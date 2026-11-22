"""Import photos of handwritten work.

For each photo: straighten the page, even out lighting, drop ruled lines and
(optionally) printed text, group the handwriting into lines and words, and save
  data/sources/<doc>/words/<wid>.png   one crop per word (white = ink darkness)
  data/sources/<doc>/overview.png      page with numbered word boxes (for context)
  data/sources/<doc>/sheet-NN.png      contact sheets of word crops (for reading)
  data/labels/<doc>.json               skeleton to fill with what each word says

usage: hw_import.py PHOTO [PHOTO ...] [--printed auto|dark|none] [--keep-color] [--no-crop]
"""
import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hwcore import DATA, components, find_page, flatten_light, jsave, load_photo, two_means_threshold, warp_page  # noqa: E402

PAGE_W = 2400
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"


def font(sz):
    try:
        return ImageFont.truetype(FONT, sz)
    except OSError:
        return ImageFont.load_default()


def next_doc_id():
    ids = [int(p.name[1:4]) for p in (DATA / "sources").glob("d[0-9][0-9][0-9]") if p.is_dir()]
    return f"d{(max(ids) if ids else 0) + 1:03d}"


def deskew(bgr):
    """Rotate so ruled lines / text lines are horizontal. Returns (bgr, degrees)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    dark = 255 - flatten_light(gray)
    m = (dark > max(30.0, cv2.threshold(dark, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0] * 0.6)).astype(np.uint8) * 255
    H, W = m.shape
    angs, wts = [], []
    segs = cv2.HoughLinesP(m, 1, np.pi / 1440, threshold=int(W * 0.08), minLineLength=int(W * 0.15),
                           maxLineGap=int(W * 0.01))
    if segs is not None:
        for x1, y1, x2, y2 in segs.reshape(-1, 4):
            a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(a) < 15:
                angs.append(a)
                wts.append(np.hypot(x2 - x1, y2 - y1))
    if len(angs) >= 5:
        o = np.argsort(angs)
        cw = np.cumsum(np.array(wts)[o])
        ang = float(np.array(angs)[o][np.searchsorted(cw, cw[-1] / 2)])  # length-weighted median
    else:  # unlined paper: pick the angle whose row profile is sharpest (text lines)
        small = cv2.resize(m, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        best = (-1, 0.0)
        for a in np.arange(-12, 12.01, 0.5):
            R = cv2.getRotationMatrix2D((small.shape[1] / 2, small.shape[0] / 2), a, 1)
            r = cv2.warpAffine(small, R, (small.shape[1], small.shape[0]))
            v = float(np.var(r.sum(1)))
            if v > best[0]:
                best = (v, -a)
        ang = best[1]
    if abs(ang) < 0.2:
        return bgr, 0.0
    R = cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1)
    return cv2.warpAffine(bgr, R, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE), ang


def ink_layers(bgr, printed="auto", keep_color=False):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    dark = 255 - flatten_light(gray)
    t = max(35.0, cv2.threshold(dark, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])
    mask = dark > t

    # Coloured marks (blue rules, red margins) are dropped when the writing itself is neutral.
    sat = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 1]
    if not keep_color and mask.any() and (sat[mask] > 70).mean() < 0.35:
        mask &= ~(sat > 70)

    # Long straight rules (lined paper, table borders). The page is deskewed first, so rules are
    # horizontal: an opening with a long 1-px kernel keeps only truly continuous lines (a row of
    # handwriting is never one unbroken stroke that long).
    H, W = mask.shape
    m8 = mask.astype(np.uint8) * 255
    vl = cv2.morphologyEx(m8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, H // 12)))
    hl = cv2.morphologyEx(m8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (W // 14, 1)))
    rules = cv2.dilate(hl | vl, np.ones((3, 3), np.uint8)) > 0
    if rules.any():
        # Keep writing that crosses a rule: only drop rule pixels about as light as the rule itself.
        rule_core = np.percentile(dark[rules & mask], 60) if (rules & mask).any() else 255
        mask &= ~(rules & (dark < rule_core + 30))

    # Printed text: a cluster of components much darker than the handwriting (pencil).
    printed_mask = np.zeros_like(mask)
    printed_t = None
    if printed != "none":
        # "Core" darkness (90th pct) separates toner from graphite far better than the mean,
        # because both have soft anti-aliased edges.
        comps, lab = components(mask, min_area=30)
        cores = []
        for c in comps:
            sl = (slice(c["y0"], c["y1"]), slice(c["x0"], c["x1"]))
            cores.append(np.percentile(dark[sl][lab[sl] == c["i"]], 90))
        cores = np.array(cores)
        thr = two_means_threshold(cores) if len(cores) >= 12 else None
        if thr is not None:
            lo, hi = cores[cores < thr], cores[cores >= thr]
            if printed == "dark" or (len(lo) >= 5 and len(hi) >= 8 and hi.mean() - lo.mean() > 25):
                printed_t = float(thr)
        if printed_t is not None:
            # Print is also small and uniform; heavy pencil can be just as dark but is bigger.
            hp = np.median([c["y1"] - c["y0"] for c, core in zip(comps, cores) if core >= printed_t])
            rm = np.zeros_like(mask)
            for c, core in zip(comps, cores):
                if core >= printed_t and (c["y1"] - c["y0"]) <= 1.45 * hp:
                    sl = (slice(c["y0"], c["y1"]), slice(c["x0"], c["x1"]))
                    rm[sl] |= lab[sl] == c["i"]
            printed_mask = cv2.dilate(rm.astype(np.uint8),
                                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
            mask &= ~printed_mask
    return dark, mask, printed_mask, printed_t


def group_words(mask):
    comps, lab = components(mask, min_area=10)
    if not comps:
        return [], lab, 0
    hs = np.array([c["y1"] - c["y0"] for c in comps])
    areas = np.array([c["area"] for c in comps])
    h = float(np.median(hs[areas >= np.median(areas)])) if len(comps) > 3 else float(hs.max())
    big = [c for c in comps if (c["y1"] - c["y0"]) >= 0.3 * h]
    small = [c for c in comps if (c["y1"] - c["y0"]) < 0.3 * h]

    lines = []
    for c in sorted(big, key=lambda c: c["x0"]):
        best, bov = None, 0.0
        for L in lines:
            if c["x0"] - L["x1"] > 8 * h:
                continue
            last = L["comps"][-5:]
            ry0 = np.median([k["y0"] for k in last])
            ry1 = np.median([k["y1"] for k in last])
            ov = (min(c["y1"], ry1) - max(c["y0"], ry0)) / max(1.0, min(c["y1"] - c["y0"], ry1 - ry0))
            if ov > 0.35 and ov > bov:
                best, bov = L, ov
        if best is None:
            lines.append(dict(comps=[c], x0=c["x0"], x1=c["x1"]))
        else:
            best["comps"].append(c)
            best["x1"] = max(best["x1"], c["x1"])
    for c in small:  # dots, commas, apostrophes: attach to the closest line that covers them
        cx, cy = (c["x0"] + c["x1"]) / 2, (c["y0"] + c["y1"]) / 2
        best, bd = None, 1.3 * h
        for L in lines:
            if L["x0"] - h <= cx <= L["x1"] + h:
                y0 = np.median([k["y0"] for k in L["comps"]])
                y1 = np.median([k["y1"] for k in L["comps"]])
                d = 0 if y0 <= cy <= y1 else min(abs(cy - y0), abs(cy - y1))
                if d < bd:
                    best, bd = L, d
        if best is not None:
            best["comps"].append(c)

    # Gaps are measured relative to each line's own letter height, so big handwriting and
    # small print on the same page don't distort each other's word spacing.
    allg = []
    for L in lines:
        L["comps"].sort(key=lambda c: c["x0"])
        hb = [c["y1"] - c["y0"] for c in L["comps"] if (c["y1"] - c["y0"]) >= 0.3 * h]
        L["h"] = float(np.median(hb)) if hb else h
        x1, L["gaps"] = None, []
        for c in L["comps"]:
            if x1 is not None:
                L["gaps"].append(max(0.0, c["x0"] - x1) / L["h"])
            x1 = c["x1"] if x1 is None else max(x1, c["x1"])
        allg += L["gaps"]
    thr_all = two_means_threshold(allg, lo=0.28, hi=1.3) or 0.55

    lines.sort(key=lambda L: np.median([(k["y0"] + k["y1"]) / 2 for k in L["comps"]]))
    words = []
    for li, L in enumerate(lines):
        thr = thr_all
        if len(L["gaps"]) >= 8:
            t2 = two_means_threshold(L["gaps"], lo=0.28, hi=1.3)
            if t2 is not None:
                thr = 0.5 * (thr_all + t2)
        cur, x1 = [], None
        for c in L["comps"]:
            if cur and (c["x0"] - x1) / L["h"] >= thr:
                words.append((li, cur))
                cur = []
            cur.append(c)
            x1 = c["x1"] if len(cur) == 1 else max(x1, c["x1"])
        if cur:
            words.append((li, cur))
    return words, lab, h


def contact_sheets(doc_dir, items, h):
    """items: list of (wid, PIL L image of ink). Returns list of sheet paths."""
    SW, PAD, out, sheets = 1600, 14, [], []
    sc = 42.0 / max(h, 1)
    x = y = PAD
    row_h = 0
    sheet = Image.new("L", (SW, 2000), 255)
    dr = ImageDraw.Draw(sheet)
    f = font(17)

    def flush():
        nonlocal sheet, dr
        p = doc_dir / f"sheet-{len(sheets) + 1:02d}.png"
        sheet.crop((0, 0, SW, min(2000, y + row_h + PAD))).convert("RGB").save(p)
        sheets.append(p)
        sheet = Image.new("L", (SW, 2000), 255)
        dr = ImageDraw.Draw(sheet)

    for wid, im in items:
        w, hh = im.size
        s = min(sc, 600 / max(w, 1), 150 / max(hh, 1))
        im = im.resize((max(1, int(w * s)), max(1, int(hh * s))), Image.LANCZOS)
        cell_w, cell_h = max(im.width, 60) + PAD, im.height + 24
        if x + cell_w > SW:
            x, y, row_h = PAD, y + row_h + PAD, 0
        if y + cell_h > 2000 - PAD:
            flush()
            x, y, row_h = PAD, PAD, 0
        dr.text((x, y), wid, fill=110, font=f)
        sheet.paste(Image.eval(im, lambda v: 255 - v), (x, y + 22))
        dr.rectangle((x - 3, y + 19, x + im.width + 3, y + 22 + im.height + 3), outline=200)
        x += cell_w
        row_h = max(row_h, cell_h)
    flush()
    return sheets


def import_photo(photo, args):
    doc = next_doc_id()
    d = DATA / "sources" / doc
    (d / "words").mkdir(parents=True, exist_ok=True)
    shutil.copy2(photo, d / ("original" + Path(photo).suffix.lower()))

    bgr = load_photo(photo)
    quad = None if args.no_crop else find_page(bgr)
    if quad is not None:
        bgr = warp_page(bgr, quad)
    s = PAGE_W / bgr.shape[1]
    bgr = cv2.resize(bgr, (PAGE_W, int(bgr.shape[0] * s)), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    bgr, skew = deskew(bgr)
    dark, mask, printed_mask, printed_t = ink_layers(bgr, args.printed, args.keep_color)
    Image.fromarray(dark, "L").save(d / "ink.png")
    words, lab, h = group_words(mask)

    recs, items = [], []
    pad = int(0.25 * h) + 2
    n = 0
    for li, comps in words:
        x0 = min(c["x0"] for c in comps); y0 = min(c["y0"] for c in comps)
        x1 = max(c["x1"] for c in comps); y1 = max(c["y1"] for c in comps)
        area = sum(c["area"] for c in comps)
        if area < 40 or (y1 - y0) > 5 * h or (x1 - x0) < 0.15 * h:
            continue
        if ((y1 - y0) < 0.3 * h and (x1 - x0) > 2.5 * h) or ((x1 - x0) < 0.3 * h and (y1 - y0) > 2.5 * h):
            continue  # left-over piece of a slightly tilted rule / table line
        if area < 0.08 * (x1 - x0) * (y1 - y0) and max(x1 - x0, y1 - y0) > 3 * h:
            continue  # sparse frame/corner of a box, not writing
        n += 1
        wid = f"w{n:03d}"
        X0, Y0 = max(0, x0 - pad), max(0, y0 - pad)
        X1, Y1 = min(dark.shape[1], x1 + pad), min(dark.shape[0], y1 + pad)
        sub = np.isin(lab[Y0:Y1, X0:X1], [c["i"] for c in comps])
        im = Image.fromarray(np.where(sub, dark[Y0:Y1, X0:X1], 0).astype(np.uint8), "L")
        im.save(d / "words" / f"{wid}.png")
        items.append((wid, im))
        recs.append(dict(id=wid, line=li, bbox=[int(X0), int(Y0), int(X1), int(Y1)]))

    # Overview with numbered boxes (printed text tinted red so it is easy to tell apart).
    ov = Image.fromarray(cv2.cvtColor(255 - dark, cv2.COLOR_GRAY2RGB))
    if printed_mask.any():
        a = np.asarray(ov).copy()
        a[printed_mask] = (a[printed_mask] * 0.35 + np.array([255, 120, 120]) * 0.65).astype(np.uint8)
        ov = Image.fromarray(a)
    dr = ImageDraw.Draw(ov)
    f = font(max(18, int(h * 0.55)))
    for r in recs:
        X0, Y0, X1, Y1 = r["bbox"]
        dr.rectangle((X0, Y0, X1, Y1), outline=(30, 110, 255), width=3)
        dr.text((X0, max(0, Y0 - int(h * 0.6))), r["id"][1:], fill=(30, 110, 255), font=f)
    ov = ov.resize((1500, int(ov.height * 1500 / ov.width)), Image.LANCZOS)
    ov.save(d / "overview.png")

    sheets = contact_sheets(d, items, h)
    meta = dict(doc=doc, source=str(Path(photo).resolve()), name=Path(photo).name, page_w=PAGE_W,
                page_h=int(dark.shape[0]), est_h=h, printed_threshold=printed_t,
                cropped_to_paper=quad is not None, words=recs, sheets=[p.name for p in sheets])
    jsave(d / "meta.json", meta)
    lines = {}
    for r in recs:
        lines.setdefault(r["line"], []).append(r["id"])
    jsave(DATA / "labels" / f"{doc}.json",
          dict(doc=doc, name=Path(photo).name, neatness=None,
               note="Fill each word with exactly what is written (keep case and punctuation). "
                    "Use '#skip' for printed text, scribbles, crossed-out or unreadable words.",
               lines=[lines[k] for k in sorted(lines)], words={r["id"]: "" for r in recs}))
    print(f"{doc}  {Path(photo).name}: {len(recs)} words, {len(lines)} lines, "
          f"paper-crop={'yes' if quad is not None else 'no'}, printed-text={'removed' if printed_t else 'none found'}")
    print(f"   overview: {d / 'overview.png'}")
    for p in sheets:
        print(f"   sheet:    {p}")
    print(f"   labels:   {DATA / 'labels' / (doc + '.json')}")
    return doc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photos", nargs="+")
    ap.add_argument("--printed", choices=["auto", "dark", "none"], default="auto",
                    help="remove printed text: auto-detect, force, or never (use 'none' for pen handwriting)")
    ap.add_argument("--keep-color", action="store_true", help="keep coloured ink (blue/red pen)")
    ap.add_argument("--no-crop", action="store_true", help="photo is already a flat scan; skip paper detection")
    args = ap.parse_args()
    for p in args.photos:
        import_photo(p, args)


if __name__ == "__main__":
    main()
