"""Which of this project's data sources can this machine actually reach?

    python scripts/check_network.py

Every fetch script in this project depends on a service somewhere else, and when
one of them is unreachable the failure surfaces forty frames deep inside pandas or
urllib, pointing at the line that made the request rather than at the reason it
failed. This answers the question directly, for all five, in one run.

It separates the two failures that look identical from inside a traceback:

* **DNS** - the name will not resolve to an address. On Windows this is
  `[Errno 11001] getaddrinfo failed`.
* **Connection** - the name resolves fine, but the bytes do not flow. Firewalls,
  TLS interception by antivirus, and a proxy that is set but unreachable all land
  here, and none of them are DNS problems even though one of them *reports* itself
  as one.

That last case is the trap. If `HTTPS_PROXY` points at a host that does not exist,
Python tries to resolve **the proxy** and raises a DNS error naming nothing useful -
while `nslookup` on the real destination succeeds, because nslookup ignores proxy
settings entirely. The proxy section below exists to catch exactly that.

Nothing here is specific to one script; run it any time a fetch fails and you want
to know whether the problem is the code or the network before reading a traceback.
"""

from __future__ import annotations

import os
import platform
import socket
import ssl
import sys
import urllib.error
import urllib.request

TIMEOUT = 15

# Every external host this project talks to, and which script needs it.
ENDPOINTS = [
    ("Census Gazetteer", "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/",
     "county lat/lon - fetch_weather.py, fetch_gridmet.py"),
    ("NASS Quick Stats", "https://quickstats.nass.usda.gov/api",
     "yields, acres - fetch_nass_yields.py, fetch_rotation.py"),
    ("NASA POWER", "https://power.larc.nasa.gov/api/temporal/daily/point",
     "daily weather - fetch_weather.py"),
    ("USDA Soil Data Access", "https://SDMDataAccess.sc.egov.usda.gov",
     "NCCPI soil ratings - fetch_soil_terrain.py"),
    ("USGS Elevation", "https://epqs.nationalmap.gov/v1/json",
     "county elevation - fetch_soil_terrain.py"),
    ("gridMET (Climatology Lab)", "https://www.northwestknowledge.net",
     "4 km weather - fetch_gridmet.py"),
]

PROXY_VARS = ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
              "http_proxy", "https_proxy", "all_proxy", "no_proxy"]


def host_of(url: str) -> str:
    return url.split("//", 1)[1].split("/", 1)[0]


def check_proxies() -> bool:
    """Report proxy variables, and whether the proxy host itself resolves.

    A proxy set to an unreachable host is the single most confusing cause of
    "DNS is fine but Python says it isn't", so it is checked first and loudly.
    """
    found = {name: os.environ[name] for name in PROXY_VARS if os.environ.get(name)}
    print("PROXY ENVIRONMENT")
    if not found:
        print("  none set - Python will connect directly\n")
        return True

    healthy = True
    for name, value in found.items():
        print(f"  {name} = {value}")
        if name.lower() == "no_proxy":
            continue
        try:
            proxy_host = host_of(value) if "//" in value else value.split("/")[0]
            proxy_host = proxy_host.rsplit("@", 1)[-1].split(":")[0]
            socket.getaddrinfo(proxy_host, None)
            print(f"      proxy host {proxy_host} resolves")
        except Exception as err:                    # noqa: BLE001
            healthy = False
            print(f"      *** proxy host does NOT resolve: {err}")
            print(f"      *** THIS IS THE PROBLEM. Every request Python makes goes")
            print(f"      *** through a proxy that does not exist, and the DNS error")
            print(f"      *** it raises names the proxy, not the site you wanted.")
    print()
    return healthy


def check(name: str, url: str, used_by: str) -> str:
    host = host_of(url)
    print(f"{name}")
    print(f"  {host}  ({used_by})")

    # Step 1: DNS only.
    try:
        infos = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
        addresses = sorted({i[4][0] for i in infos})
        print(f"  DNS  ok   {', '.join(addresses[:3])}"
              + (f" (+{len(addresses) - 3} more)" if len(addresses) > 3 else ""))
    except Exception as err:                        # noqa: BLE001
        print(f"  DNS  FAIL {err}")
        print("       name lookup failed - check the proxy section above first\n")
        return "dns"

    # Step 2: an actual HTTPS request. Resolving is not the same as connecting.
    request = urllib.request.Request(url, headers={"User-Agent": "yieldpred-netcheck"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            print(f"  HTTP ok   {response.status}\n")
            return "ok"
    except urllib.error.HTTPError as err:
        # A 403/404 still proves the connection works, which is what we are testing.
        print(f"  HTTP ok   reached it (HTTP {err.code} - the host answered)\n")
        return "ok"
    except ssl.SSLError as err:
        print(f"  TLS  FAIL {err}")
        print("       the host answered but the secure handshake failed - usually")
        print("       antivirus or a security suite intercepting HTTPS\n")
        return "tls"
    except Exception as err:                        # noqa: BLE001
        print(f"  HTTP FAIL {err}")
        print("       resolved but could not connect - firewall, or the service is down\n")
        return "conn"


def main() -> None:
    print("=" * 70)
    print("CAN THIS MACHINE REACH THE PROJECT'S DATA SOURCES?")
    print("=" * 70)
    print(f"python {platform.python_version()} on {platform.system()} "
          f"{platform.release()}   {ssl.OPENSSL_VERSION}\n")

    proxies_ok = check_proxies()
    results = {name: check(name, url, used) for name, url, used in ENDPOINTS}

    print("=" * 70)
    reachable = [n for n, r in results.items() if r == "ok"]
    print(f"{len(reachable)} of {len(results)} reachable")
    for name, result in results.items():
        if result != "ok":
            print(f"  FAILED  {name}  ({result})")

    if not proxies_ok:
        print("\nFix the proxy variables first - they explain every failure above.")
        print("To clear them for one PowerShell session:")
        print("  $env:HTTP_PROXY=''; $env:HTTPS_PROXY=''")
    elif all(r == "dns" for r in results.values()):
        print("\nEvery name failed to resolve, so this is not about any one service.")
        print("Check VPN, DNS settings, or whether the connection is up at all.")
    elif len(reachable) == len(results):
        print("\nEverything is reachable. A fetch failing now is the code, not the network.")

    sys.exit(0 if len(reachable) == len(results) else 1)


if __name__ == "__main__":
    main()
