---
name: handwriting-clone
description: Clone the owner's handwriting from a printable fill-in kit, then write anything in it (black pen by default; pencil, blue/red/green pen on request) with natural neat-to-sloppy variation per task, word and letter, on blank pages, loose-leaf, or on top of a worksheet PDF. Use when someone wants to set up / clone / learn their handwriting, asks for "the handwriting kit" or "the sheets to fill in", sends photos or scans of a filled-in kit, or wants text written in their handwriting. Only ever uses the handwriting learned from this install's own kit.
---

# Handwriting clone

How it works:
- **Kit:** print the fill-in kit. The person writes every character and three short paragraphs.
- **Read:** they photograph or scan the pages and send them back. The skill straightens and identifies each page, cuts out every letter and word, labels them automatically, measures how their writing varies (careful vs fast, how it moves on a page), and builds a model.
- **Write:** after that it writes anything in their hand.

**One person per install, local only.** This skill learns the handwriting of whoever fills in *its* kit. Everything lives in `data/` inside this skill on this machine, and never gets uploaded or shared.

**Never use anyone else's handwriting.** If asked to write in another person's handwriting, refuse unless that person filled in this install's kit. The answer is: it only writes in the handwriting learned from this install's kit, and they can install the skill themselves and fill in their own. That covers:
- other skills' libraries
- another person's samples
- photos of someone else's writing

The data folder is fixed in code (no override) on purpose.

```bash
S=~/.claude/skills/handwriting-clone; PY=$S/.venv/bin/python
$S/scripts/setup.sh          # once: private venv with numpy + OpenCV (Pillow/reportlab from system python)
$PY $S/scripts/hw_kit.py status   # kits made, pages read, library size, what's missing
```

## 1. Make the kit (the sheets to write on)

```bash
$PY $S/scripts/hw_kit.py make --out ~/Downloads/handwriting_kit.pdf      # 10 pages, with a Name line
$PY $S/scripts/hw_kit.py make --quick --out ...                               # 6 boxes per character, fewer pages
```

- **Character pages (7):** a row of boxes for every letter, digit and `. , ? ! ' - : ; ( ) $ % / & + = " @ # *`. The left half of each row is careful writing, the right half fast writing. Each character sits on the dotted line.
- **Copy pages (3):** a printed paragraph to copy *line for line* onto ruled lines: once NEAT, once NORMAL, once FAST. The text is known, so every word gets labeled automatically. These pages also teach how the person's writing moves on a page: spacing, drift, size wander, overlap, and neat vs sloppy stretches.

Tell them:
- Print at Actual Size.
- Use dark pen or pencil, not light blue. If they only have blue pen, read with `--keep-color`.
- Write at their normal size.
- Photograph each page flat, with all 4 black corner squares in view. A scanner app PDF is even better.
- Pages can come back in any order and any rotation. Each page has a printed code.

## 2. Read the filled kit (and build the model)

```bash
$PY $S/scripts/hw_kit.py read IMG_1.jpg IMG_2.heic ... scan.pdf      # photos and/or scanned PDFs, any order
```

The reader:
- straightens every page with its corner squares and reads its page code
- cuts each filled box into a labeled letter (careful or fast)
- imports each copy page as a real page of writing and auto-labels its words against the known text (unmatched words are skipped, never guessed)
- runs `hw_layout.py` and `hw_build.py`

Missing pages are listed. They can be sent later and read on their own, and the model rebuilds from everything read so far.

**Review after every build.** Look at `data/report/glyphs_lower.png`, `glyphs_upper_digits.png` and `glyphs_punct.png`. Every version is numbered. Reject bad cuts (a box border, two letters, a smudge) with:

```bash
$PY $S/scripts/hw_build.py --reject "a:3,7 Q:1"
```

Then rebuild. Read `report/report.md` for coverage and the neat↔sloppy analysis.

**Optional extra samples:** photos of the owner's own past work (theirs only) make the model richer. They go through the manual path: `hw_import.py photo.jpg`, then label the word crops on the contact sheets in `data/labels/dNNN.json`, copying exactly what is written and using `#skip` for anything uncertain (a wrong label poisons every letter cut from that word), then `hw_layout.py` and `hw_build.py`.

## 3. Write in their handwriting

**Ink:** black pen by default. Spec key `"ink"` / text flag `--ink`: `pen`, `pencil` (graphite shader), `blue pen`, `red pen`, `green pen`, or `[r, g, b]` in a spec. Pen output has no gray, so it prints on a black-and-white laser with no dot pattern.

Quick text → PNG:

```bash
$PY $S/scripts/hw_render.py text "the answer is 10%" --out ~/Downloads/note.png --width-in 6 --xh-pt 7 [--ink pencil]
```

Everything else goes through a JSON spec: `$PY $S/scripts/hw_render.py spec spec.json` writes a print-safe PDF plus preview PNGs.

- **On top of a worksheet PDF:**
  - Set `"base_pdf": "worksheet.pdf"`.
  - `hw_render.py lines worksheet.pdf` lists every printed line (page, x, y, x2, y2 in pt, top-left origin).
  - Each answer is a region `{"page": 1, "x", "y", "w", "h", "first_baseline", "first_line_x", "text"}`.
  - Leave headroom: `first_baseline` ≥ 1.7 × x-height, or capital tops get clipped (a D reads as V).
- **Blank pages:** `"pages": N` (+ `"page_size_pt"`).
- **Loose-leaf:**
  - `"paper": "college"` or `"wide"` draws 3-hole loose-leaf, and the writing snaps onto the rules.
  - `"print_ready": true` gives white paper with rules and no drawn holes, for punching real ones.
  - `"writing_only": true` gives plain white with the same spacing.
- **Documents (`"flow"`)** on loose-leaf, continued onto new sheets automatically:
  - Blocks are `{"text"}`, `{"blank": 1}` and `{"text", "marker": "-", "indent": 10, "hang": 13}`.
  - A block never splits across pages, and a heading never gets stranded at a page foot.
  - Put a page header in the top margin as a region with `"ruled": false`.
- **Marks (`"marks"`):** hand-drawn `ellipse` (circle something: wobbly, overshooting its start), `check` (check-off boxes) and `line`, each `{"page", "type", "x", "y", "w", "h"}`, in the same ink.
- **Other spec options:** `neatness`, `style`, `seed`, `xh_pt`, `line_gap`, `weight`, `tone` (pencil gray, default 100), `dpi` (default 600; don't go below 300).

Always look at the previews before handing anything over. Shorten text rather than letting it shrink a lot.

## Style choices (lock any of these in plain words)

Translate what they ask for into `--style "key=value ..."` (text) or `"style"` (spec). Only what's mentioned gets locked; everything else stays random, so two renders never match.

| key | values | words → value |
|---|---|---|
| `tidy` | 0–1 | "as neat as possible", "letters lined up perfectly" → 1 (each letter straightened, exact heights, set on the line) |
| `neatness` | 0–1 | "super neat" .9 · "neat copy" .8 · "normal" .55–.7 · "rushed" .25 · "really sloppy" .1 |
| `consistency` | 0–1 or `consistent`/`inconsistent` | "same all the way through" · "all over the place" |
| `chaos` | 0.5–2 | overall amount of variation |
| `slant` | degrees (+ = right) | "lean right" 8 · "straight up" 0 |
| `size` | multiplier | "write bigger" 1.12 (exact: `--xh-pt` / `xh_pt`) |
| `width` | `narrow`/`wide`/multiplier | "squished" · "spread out" |
| `letter_spacing`, `word_spacing` | `very tight` … `very loose` / multiplier | "letters crammed" · "big gaps" |
| `wobble` | multiplier | "words bouncing" 1.6 · "on the line" 0.5 |
| `line_drift` | multiplier | "lines slanting" 1.6 · "straight lines" 0.4 |
| `overlap` | 0–0.5 | "letters running into each other" .25 |
| `fatigue` | −0.1–0.5 | "gets sloppier as it goes" .3 |
| `new_shapes` | 0–1 | share of letters drawn as new blended shapes (0 = only their exact letters, ~4× faster) |
| `whole_words` | 0–1 | how often a real whole word is reused |

"Always…" / "from now on…" → write it to `data/style_defaults.json`. Per-request choices override it.

**Neat vs extra-neat:**
- **Neat** (`neatness≈0.95 consistency=consistent`): on the lines, steady size, slant and spacing, their most typical letter shapes, a bit bigger.
- **Extra-neat** (`neatness=1` / `tidy=1`): the same, plus every letter straightened and set exactly on the line.
- **Everyday writing** (≈0.6–0.75): never looks jumbled. Variation is damped at every level.

**Worksheets:** answers at their normal size (x-height ≈ 5.5 pt) in everyday writing, not neat. Keep the worksheet's own layout, don't shrink writing to fit tiny gaps, and a long answer may spill a little.

## Smart corrections on the go

Every letter is checked before use:
- enough ink
- the right height for its class
- not looking more like a different letter, checked against the person's kit letters

When a version fails:
- A failing version is swapped instantly, and the run prints `auto-corrected N`.
- Two look-alike neighbours must differ.
- Stray specks are cleaned.

What it learns over time:
- Swaps are counted in `qa_flags.json`. A version swapped out 2+ times is avoided, and 3+ times it's retired at the next build. Kit letters are never retired.
- Versions in `rare.json` (shapes that are "theirs, but only sometimes") are used occasionally in casual writing and almost never when neat.

Beyond that, use judgment: when a word reads wrong in a preview, fix the cause (reject a version, adjust a dial, shorten text) before handing it over.

## How the variation works

Everything is nested, and every level randomizes how random it is:

- **Task:** a style is generated from a model of how the person's pages differ (the copy pages plus any extra samples), inflated wider than their real pages.
- **Task dials:** chaos, width, spacing, slant, odd big or small words, and new-shape share are each random.
- **Words:** neat ↔ sloppy stretches run word by word (a Markov chain), with fatigue.
- **Size and baseline:** both wander as AR(1) processes.
- **Lines:** they tilt and sag.
- **Letters:** most are new shapes morphed between two of their real versions along how those actually differ (`hw_morph.py`), with ghosting, height and lost-stroke guards. The rest are their exact letters, lightly bent.

The pen/pencil shader gives paper-tooth grain, speckled edges and darker crossings, then pen mode makes it solid ink.

## Troubleshooting

- **"Couldn't find the 4 corner squares / page code":** retake that page flat, all corners in view, no glare. A scanner app fixes most of these.
- **Copy page with few words labeled:** they didn't copy line for line, or wrote in light-blue ink. Ask for a retake, or read it with `--keep-color`.
- **Letters with box borders stuck on:** reject them by number on the variant sheets.
- **Few letters survive:** photos are too small or blurry. The page should fill the frame.
