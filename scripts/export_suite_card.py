"""Write this app's card for the farm-account home page: every county's trend corn yield.

    python scripts/export_suite_card.py

Reads data/processed/*_corn_yield_county.csv and writes data/processed/suite_card.json.
Run it after new NASS yields are fetched (once a year), then in farm-account:
    python -m farm_account.cards refresh
See src/yieldpred/suite_card.py for the method.
"""

from pathlib import Path

from yieldpred import suite_card

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def main() -> None:
    paths = sorted(PROCESSED.glob("*_corn_yield_county.csv"))
    data = suite_card.build(paths)
    out = PROCESSED / "suite_card.json"
    suite_card.write(data, out)
    print(f"Wrote {out.name}: {len(data['cards'])} counties for the {data['season']} season "
          f"({len(data['skipped'])} skipped: too few or no recent yields).")


if __name__ == "__main__":
    main()
