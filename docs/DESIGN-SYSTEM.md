# Sky & Soil — Design System

The shared visual language for this app and the others alongside it. It is committed
rather than kept in working notes, because the point of a design system is that a
second app can copy it.

**Two files carry the whole thing:**

| File | What it controls |
|---|---|
| `src/yieldpred/theme.py` | every colour used by a chart or map, in one place |
| `.streamlit/config.toml` | the app chrome — buttons, sidebar, headings, widgets |

Copy both into a new app and it will look like a sibling on the first run.

## The Palette

| Role | Hex | Name |
|---|---|---|
| Primary | `#1A5B96` | plains blue |
| Secondary | `#A8622D` | turned earth |
| Accent | `#A8861E` | ripe wheat |
| Ink | `#1C2024` | |
| Chart surface | `#FBFAF8` | warm paper |
| Page plane | `#F4F2EE` | |

Warm paper rather than white is deliberate: pure `#FFFFFF` next to saturated marks
reads as glare, and a slightly warm surface makes the earth pole of the diverging
scale sit naturally rather than looking like a stain.

### Categorical — six slots, fixed order

```
1 #1A5B96  plains blue     4 #A8861E  ripe wheat
2 #A8622D  turned earth    5 #D46A8B  rose
3 #1B8F7A  teal            6 #5A4BA8  violet
```

**The order is the accessibility mechanism, not a style choice.** Assign slot 1 to
the first series, slot 2 to the second, and never cycle or re-sort by rank — a
filter that removes a series must not repaint the survivors, or the reader has to
re-learn the colours every time they touch a control.

**There is no seventh slot.** Past six, fold the tail into "Other" or use small
multiples. `theme.series_colors()` raises rather than inventing a hue, because a
generated seventh colour is indistinguishable from one of the first six for a
meaningful share of readers.

### Sequential — one hue, light to dark

```
#D6E4F2  #AFC9E4  #85A9D2  #5A88BF  #3670A8  #1A5B96  #0F4273
```

Magnitude has an order, and a single hue is the only encoding a reader interprets
as ordered without being told. Never a rainbow: viewers read rainbow bands as
unordered categories, which is exactly wrong for a quantity.

### Diverging — two hues with a neutral midpoint

```
#1A5B96  #6D9BC4  #EDEAE4  #D2A176  #A8622D
```

Used only where zero means something — prediction error does, raw yield doesn't.
**The domain must be symmetric** (±the largest absolute value) so the neutral step
lands exactly on zero. An asymmetric diverging scale silently misreports which
places are above and below.

Blue↔earth survives the common forms of colour blindness. Red↔green does not, which
is why it never appears here despite being the obvious choice for "good and bad".

### Status — reserved

```
good #1F7A33   warning #D39E00   serious #C25A22   critical #B3332F
```

A status colour never doubles as "series 7", and never carries meaning alone — it
ships with an icon or a word beside it.

## The Palette Is Validated, Not Chosen

This is the part worth copying into every future project.

Colour-blind safety is **computable**, so it was computed rather than eyeballed.
Every slot was checked against the actual chart surface, in both light and dark mode,
for:

- **Lightness band** — all slots inside the mode's OKLCH lightness range
- **Chroma floor** — above the point where a hue reads as grey
- **CVD separation** — OKLab ΔE between adjacent slots under simulated protanopia
  and deuteranopia
- **Normal-vision floor** — full-colour readers must be able to separate neighbours too
- **Contrast** — every slot ≥ 3:1 against the surface it renders on

Results: **all six slots pass every check in both light and dark mode**, and the
first three additionally clear the harder *all-pairs* gate that scatter plots and
choropleths need (where any two series can end up adjacent, not just neighbours in
a legend).

Getting there took iteration, which is the point. The first plains blue failed the
chroma floor at 0.092 — it read as grey-blue rather than blue — and the first wheat
sat at 2.32:1 contrast. Both were re-stepped until they passed. **Neither failure
was visible to me by eye.**

To re-run after any change, use the validator from the `dataviz` skill:

```bash
node validate_palette.js "#1A5B96,#A8622D,#1B8F7A,#A8861E,#D46A8B,#5A4BA8" \
  --mode light --surface "#FBFAF8"
node validate_palette.js "#4A90D9,#C5722F,#1F9E7E,#B78F1D,#D26C8F,#8B7CDC" \
  --mode dark --surface "#15181B"
```

## Rules That Apply To Every Chart

1. **One y-axis.** Never two scales on one chart. Two measures of different
   magnitude get two charts, small multiples, or indexing to a common base.
2. **Colour follows the entity, never its rank.**
3. **A legend whenever there are two or more series**, so identity is never carried
   by colour alone; one series needs no legend, because the title names it.
4. **Text wears text colours**, never the series colour. A coloured mark beside the
   label carries the identity.
5. **The data is the loudest thing on the page.** Gridlines and axes recede;
   boundaries are thin; the title states the finding rather than the variable name.
6. **Render it and look at it.** The validator checks colour, not layout.

Rule 6 earned its place on the first run of this system — see below.

## No Data Is Not Zero

Two bugs surfaced the moment a map was actually rendered and examined, and both are
worth knowing about because neither produces an error message.

**The near-collision.** The first "no data" grey was `#ECE9E3` and the diverging
midpoint is `#EDEAE4`. A county with no data was very nearly the same colour as a
county the model predicted perfectly — two opposite meanings, one appearance. No
data now uses `#D5CFC5` *and* a hatch, so the distinction survives greyscale
printing and colour-blind vision as well.

**The vanishing counties.** Grant and Hooker are Sandhills counties that grow almost
no corn, so NASS never reports them. `align_to_geometry` used an inner join, so they
were dropped before rendering — and a dropped polygon doesn't leave a gap a reader
can interpret, it leaves *nothing*, with the page showing through. They had been
missing from every figure in the project.

The fix distinguishes two jobs that look identical in code:

- **Statistics use an inner join.** Moran's I cannot compute a contiguity-weighted
  average where a neighbour has no value; those counties must leave the graph.
- **Maps use a left join.** Every polygon is drawn, and the ones without data are
  marked as such.

`align_to_geometry(..., how=...)` now takes the join explicitly, and the reason is
documented at the call site rather than in someone's memory.

## Using It In A New App

1. Copy `src/<pkg>/theme.py` and `.streamlit/config.toml`.
2. Import colours from the theme module. Never write a hex value into a chart.
3. If the new app needs a hue this palette doesn't have, add it to the theme and
   **re-run the validator on the whole set** — a palette is only safe as a set.
4. Keep the six rules above. They're not house style; they're the difference
   between a chart that can be read and one that merely looks finished.
