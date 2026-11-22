"""Measure how the person's writing moves on the page, from the original photos (no labels needed).

For every imported page: straighten, find the lines of writing in the main text block and
measure, in x-heights:
  line drift (slope), line curvature (sag/hump), words bobbing off the line (and how
  correlated neighbours are), word size wander along a line, letter gaps incl. overlaps,
  word gaps, line spacing, margin creep, how wobble changes line to line (neat/sloppy
  stretches + how sticky they are) and down the page (fatigue).
Writes data/layout.json; hw_build.py folds it into profile.json["layout"].

usage: hw_layout.py [--doc d001 ...]
"""
import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hw_build import deslant  # noqa: E402
from hw_import import PAGE_W, deskew, group_words, ink_layers  # noqa: E402
from hwcore import DATA, jload, jsave, load_photo  # noqa: E402

QS = [0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95]  # stored quantiles of empirical distributions


def ar1(x):
    x = np.asarray(x, float)
    if len(x) < 4 or x.std() < 1e-9:
        return None
    x = x - x.mean()
    return float(np.clip((x[:-1] * x[1:]).sum() / (x * x).sum(), -0.9, 0.95))


def q(v):
    v = np.asarray(v, float)
    return [round(float(np.quantile(v, p)), 4) for p in QS] if len(v) >= 8 else None


def analyse_page(photo, dark_cut=None):
    bgr = load_photo(photo)
    s = PAGE_W / bgr.shape[1]
    bgr = cv2.resize(bgr, (PAGE_W, int(bgr.shape[0] * s)), interpolation=cv2.INTER_AREA)
    bgr, _ = deskew(bgr)
    dark, mask, _, _ = ink_layers(bgr, "none", False)
    words, lab, h = group_words(mask)

    # ---- per word geometry
    lines = {}
    for li, comps in words:
        big = [c for c in comps if (c["y1"] - c["y0"]) >= 0.45 * h]
        if not big:
            continue
        x0 = min(c["x0"] for c in comps); x1 = max(c["x1"] for c in comps)
        y0 = min(c["y0"] for c in comps); y1 = max(c["y1"] for c in comps)
        area = sum(c["area"] for c in comps)
        fill = area / max(1, (x1 - x0) * (y1 - y0))
        if fill < 0.04 or fill > 0.55 or (y1 - y0) > 4 * h:
            continue  # speckle texture, solid blobs, drawings
        xh = float(np.median([c["y1"] - c["y0"] for c in big]))
        base = float(np.percentile([c["y1"] for c in big], 65))
        cs = sorted(comps, key=lambda c: c["x0"])
        gaps, xr = [], None
        for c in cs:
            if xr is not None and (c["y1"] - c["y0"]) >= 0.3 * h:
                gaps.append((c["x0"] - xr) / xh)
            if (c["y1"] - c["y0"]) >= 0.3 * h:
                xr = c["x1"] if xr is None else max(xr, c["x1"])
        lines.setdefault(li, []).append(dict(x0=x0, x1=x1, xc=(x0 + x1) / 2, base=base, xh=xh, gaps=gaps,
                                             box=(x0, y0, x1, y1), ids=[c["i"] for c in comps]))

    # ---- keep the main text block: lines with >= 3 plausible words of typical size
    med = np.median([w["xh"] for ws in lines.values() for w in ws]) if lines else h
    good = []
    for li, ws in lines.items():
        ws = [w for w in ws if 0.5 * med <= w["xh"] <= 2.2 * med]
        ws.sort(key=lambda w: w["x0"])
        if len(ws) >= 3 and (ws[-1]["x1"] - ws[0]["x0"]) > 0.15 * PAGE_W:
            good.append(ws)
    good.sort(key=lambda ws: np.median([w["base"] for w in ws]))
    if len(good) < 3:
        return None

    X = float(np.median([w["xh"] for ws in good for w in ws]))
    slopes, curv, wob, wob_phi, size_sd, size_phi, wgaps, lgaps, starts, bases = [], [], [], [], [], [], [], [], [], []
    slants = []
    for ws in good:
        xs = np.array([w["xc"] for w in ws]); ys = np.array([w["base"] for w in ws])
        k, b = np.polyfit(xs, ys, 1)
        slopes.append(math.degrees(math.atan(k)))
        r = (ys - (k * xs + b)) / X
        wob.append(float(r.std()))
        p = ar1(r)
        if p is not None:
            wob_phi.append(p)
        if len(ws) >= 5:
            a2 = np.polyfit((xs - xs.min()) / max(1, np.ptp(xs)), ys / X, 2)[0]
            curv.append(float(a2) / 4)  # sag of the line's middle, in x-heights
        ls = np.log([w["xh"] / X for w in ws])
        size_sd.append(float(np.std(ls)))
        p = ar1(ls)
        if p is not None:
            size_phi.append(p)
        for A, B in zip(ws[:-1], ws[1:]):
            wgaps.append((B["x0"] - A["x1"]) / X)
        for w in ws:
            lgaps += w["gaps"]
        starts.append(ws[0]["x0"] / X)
        bases.append(float(np.median(ys)))
        for w in ws[: min(6, len(ws))]:  # slant from a few words per line (deslant is slow)
            x0, y0, x1, y1 = w["box"]
            sub = np.where(np.isin(lab[y0:y1, x0:x1], w["ids"]), dark[y0:y1, x0:x1], 0).astype(np.uint8)
            if sub.size:
                slants.append(deslant(sub)[1])
    spacing = np.diff(bases) / X
    spacing = spacing[(spacing > 1.2) & (spacing < 8)]

    wob = np.array(wob)
    thr = np.median(wob) * 1.35
    state = (wob > thr).astype(int)                      # 1 = a sloppier line than this page's norm
    switches = float(np.mean(state[1:] != state[:-1])) if len(state) > 2 else None
    third = max(1, len(wob) // 3)
    fatigue = float((wob[-third:].mean() - wob[:third].mean()) / max(1e-6, wob.mean()))
    lg = np.array(lgaps)
    return dict(
        n_lines=len(good), n_words=int(sum(len(ws) for ws in good)), xh_px=round(X, 2),
        slope_mean=round(float(np.mean(slopes)), 3), slope_sd=round(float(np.std(slopes)), 3),
        curv_sd=round(float(np.std(curv)), 4) if len(curv) >= 3 else None,
        wobble_mean=round(float(wob.mean()), 4), wobble_cv=round(float(wob.std() / max(1e-6, wob.mean())), 4),
        wobble_phi=round(float(np.median(wob_phi)), 3) if wob_phi else None,
        size_sd=round(float(np.median(size_sd)), 4), size_phi=round(float(np.median(size_phi)), 3) if size_phi else None,
        word_gap_q=q(wgaps), letter_gap_q=q(lgaps),
        overlap_rate=round(float((lg < 0).mean()), 4) if lg.size else None,
        overlap_depth=round(float(-np.median(lg[lg < 0])), 4) if (lg < 0).any() else None,
        line_spacing_mean=round(float(spacing.mean()), 3) if spacing.size >= 2 else None,
        line_spacing_cv=round(float(spacing.std() / spacing.mean()), 4) if spacing.size >= 2 else None,
        margin_sd=round(float(np.std(starts)), 3),
        slant_mean=round(float(np.mean(slants)), 2) if slants else None,
        slant_sd=round(float(np.std(slants)), 2) if slants else None,
        sloppy_line_share=round(float(state.mean()), 3), switch_rate=round(switches, 3) if switches is not None else None,
        fatigue=round(fatigue, 3))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--doc", nargs="*")
    a = ap.parse_args()
    out = jload(DATA / "layout.json", {})
    for d in sorted((DATA / "sources").glob("d[0-9][0-9][0-9]")):
        if a.doc and d.name not in a.doc:
            continue
        orig = next(iter(sorted(d.glob("original.*"))), None)
        if orig is None:
            continue
        st = analyse_page(orig)
        if st is None:
            print(f"{d.name}: not enough clean lines")
            continue
        st["name"] = (jload(d / "meta.json") or {}).get("name")
        out[d.name] = st
        print(f"{d.name} {st['name']}: {st['n_lines']} lines/{st['n_words']} words  slope±{st['slope_sd']}°  "
              f"wobble {st['wobble_mean']} (cv {st['wobble_cv']})  overlap {st['overlap_rate']}  "
              f"sloppy lines {st['sloppy_line_share']} switch {st['switch_rate']}  fatigue {st['fatigue']:+}")
    jsave(DATA / "layout.json", out)


if __name__ == "__main__":
    main()
