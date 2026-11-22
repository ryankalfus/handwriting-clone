"""Build the handwriting library + analysis profile from labeled imports.

Reads data/labels/*.json (filled in by reading the contact sheets) and writes
  data/library/glyphs/<uXXXX>/*.png   every usable letter, upright, white = ink
  data/library/words/*.png            whole words (reused verbatim sometimes)
  data/profile.json                   stats + neat/sloppy model + glyph/word index
  data/report/report.md (+ PNGs)      human-readable analysis

usage: hw_build.py [--min-conf 0.55]
"""
import argparse
import math
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hwcore import (ASC, CAPS, DATA, DESC, DIGITS, ON_BASE, XH, char_dir, components, jload, jsave,  # noqa: E402
                    rel_w)

# Variability features: (value when very neat, value when very sloppy). Used to score
# each document's neatness on an absolute scale (works even with a single document).
VAR_REF = {
    "xh_cv": (0.05, 0.22),          # word-to-word x-height variation (coef. of variation)
    "size_ratio_sd": (0.05, 0.22),  # letter-to-letter size variation inside words (x-height units)
    "base_resid_sd": (0.04, 0.18),  # letters bouncing off the word baseline (x-height units)
    "slant_sd": (1.5, 8.0),         # word-to-word slant variation (degrees)
    "letter_gap_sd": (0.06, 0.30),  # irregular spacing between letters (x-height units)
    "word_gap_sd": (0.12, 0.50),    # irregular spacing between words
    "word_base_sd": (0.04, 0.25),   # words drifting off the line
    "line_slope_sd": (0.3, 2.5),    # lines tilting differently (degrees)
}
MEAN_DEFAULTS = {"slant_mean": 4.0, "letter_gap_mean": 0.15, "word_gap_mean": 0.95, "line_slope_mean": 0.0,
                 "asc_ratio": 1.8, "cap_ratio": 1.65, "digit_ratio": 1.6, "desc_depth": 0.75,
                 "stroke": 0.13, "ink_core": 150.0}
WANTED = (list("abcdefghijklmnopqrstuvwxyz") + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + list("0123456789")
          + list(".,?!'-:;()$%/&+=\""))


# ------------------------------------------------------------------ segmentation
def deslant(alpha, fixed=None):
    """Shear the word upright. Returns (alpha_upright, slant_deg); +deg = leans right.
    fixed: use this angle instead of searching (single letters are unreliable to search)."""
    m = (alpha > 30).astype(np.float32)
    h, w = m.shape
    cy = h / 2
    best = (-1, 0.0 if fixed is None else float(fixed))
    for deg in ([] if fixed is not None else range(-30, 31, 2)):
        k = math.tan(math.radians(deg))
        pad = int(abs(k) * h) + 2
        M = np.float32([[1, k, -k * cy + pad], [0, 1, 0]])
        s = cv2.warpAffine(m, M, (w + 2 * pad, h), flags=cv2.INTER_NEAREST)
        score = float((s.sum(0) ** 2).sum())
        if score > best[0]:
            best = (score, deg)
    deg = best[1]
    k = math.tan(math.radians(deg))
    pad = int(abs(k) * h) + 2
    M = np.float32([[1, k, -k * cy + pad], [0, 1, 0]])
    up = cv2.warpAffine(alpha, M, (w + 2 * pad, h), flags=cv2.INTER_LINEAR)
    return up, float(deg)


def units_from(alpha, est_h):
    mask = alpha > 30
    comps, lab = components(mask, 4)
    # drop rule-line remnants: long and very thin
    comps = [c for c in comps if not ((c["y1"] - c["y0"]) < max(3, 0.1 * est_h) and (c["x1"] - c["x0"]) > 1.2 * est_h)]
    if not comps:
        return [], lab
    hs = np.array([c["y1"] - c["y0"] for c in comps])
    areas = np.array([c["area"] for c in comps])
    big_h = np.median(hs[areas >= np.median(areas)])
    base_guess = np.percentile([c["y1"] for c in comps if (c["y1"] - c["y0"]) > 0.5 * big_h], 40)
    xh_guess = 0.6 * big_h
    units = []
    for c in sorted(comps, key=lambda c: -c["area"]):
        small = (c["y1"] - c["y0"]) < 0.45 * big_h and c["area"] < 0.3 * np.median(areas)
        above = c["y1"] < base_guess - 0.55 * xh_guess
        if small and above:  # i/j dots, accents: merge into the letter below
            cx = (c["x0"] + c["x1"]) / 2
            host = min((u for u in units if u["x0"] - 0.3 * xh_guess <= cx <= u["x1"] + 0.3 * xh_guess),
                       key=lambda u: abs((u["x0"] + u["x1"]) / 2 - cx), default=None)
            if host is not None:
                host["ids"].append(c["i"])
                host["x0"], host["x1"] = min(host["x0"], c["x0"]), max(host["x1"], c["x1"])
                continue
        units.append(dict(ids=[c["i"]], x0=c["x0"], x1=c["x1"]))
    return sorted(units, key=lambda u: (u["x0"] + u["x1"]) / 2), lab


def atoms_from(units, lab, s):
    """Over-segment wide blobs at thin columns; DP regroups atoms into letters."""
    atoms = []
    for u in units:
        um = np.isin(lab, u["ids"])
        cols = um.sum(0).astype(np.float32)
        x0, x1 = u["x0"], u["x1"]
        cuts = []
        if (x1 - x0) > 1.3 * s:
            seg = np.convolve(cols[x0:x1], np.ones(3) / 3, mode="same")
            ref = np.median(seg[seg > 0]) if (seg > 0).any() else 1
            last = x0
            for x in range(x0 + int(0.3 * s), x1 - int(0.3 * s)):
                v = seg[x - x0]
                if v <= 0.45 * ref and v <= seg[max(0, x - x0 - 2):x - x0 + 3].min() + 1e-6 and x - last >= 0.3 * s:
                    cuts.append(x)
                    last = x
        edges = [x0] + cuts + [x1]
        for a, b in zip(edges[:-1], edges[1:]):
            m = um.copy()
            m[:, :a] = False
            m[:, b:] = False
            if m.sum() > 3:
                xs = np.where(m.any(0))[0]
                atoms.append(dict(mask=m, x0=int(xs[0]), x1=int(xs[-1]) + 1, split=bool(cuts)))
    return sorted(atoms, key=lambda a: (a["x0"] + a["x1"]) / 2)


def assign(atoms, chars, s):
    """Group consecutive atoms into len(chars) letters minimising width mismatch."""
    M, n = len(atoms), len(chars)
    if M < n or n == 0:
        return None
    INF = 1e18

    def cost(i, k, j):
        x0 = min(a["x0"] for a in atoms[i:k + 1]); x1 = max(a["x1"] for a in atoms[i:k + 1])
        w = max(1.0, x1 - x0)
        c = math.log(w / (s * rel_w(chars[j]))) ** 2
        for a, b in zip(atoms[i:k], atoms[i + 1:k + 1]):
            c += 0.8 * max(0.0, b["x0"] - a["x1"]) / s
        return c

    D = [[INF] * M for _ in range(n)]
    P = [[-1] * M for _ in range(n)]
    for k in range(M - n + 1):
        D[0][k] = cost(0, k, 0)
    for j in range(1, n):
        for k in range(j, M - (n - 1 - j)):
            for i in range(j, k + 1):
                if D[j - 1][i - 1] < INF:
                    v = D[j - 1][i - 1] + cost(i, k, j)
                    if v < D[j][k]:
                        D[j][k], P[j][k] = v, i
    if D[n - 1][M - 1] >= INF:
        return None
    groups, k = [], M - 1
    for j in range(n - 1, -1, -1):
        i = P[j][k] if j > 0 else 0
        groups.append((i, k, cost(i, k, j)))
        k = i - 1
    return groups[::-1]


# ------------------------------------------------------------------ per word
def analyse_word(alpha_raw, label, est_h, known_baseline=None, slant_hint=None, fixed_xh=None):
    alpha, slant = deslant(alpha_raw)
    if slant_hint is not None and abs(slant - slant_hint[0]) > slant_hint[1]:
        alpha, slant = deslant(alpha_raw, fixed=slant_hint[0])   # implausible guess: use the page's typical slant
    chars = [c for c in label if c != " "]
    units, lab = units_from(alpha, est_h)
    if not units:
        return None
    tot = max(u["x1"] for u in units) - min(u["x0"] for u in units)
    s = tot / max(0.5, sum(rel_w(c) for c in chars) + 0.15 * (len(chars) - 1))
    atoms = atoms_from(units, lab, s)
    groups = assign(atoms, chars, s)
    if groups is None:
        return dict(slant=slant, alpha=alpha, ok=False)
    letters = []
    for (i, k, c), ch in zip(groups, chars):
        m = np.zeros_like(atoms[0]["mask"])
        for a in atoms[i:k + 1]:
            m |= a["mask"]
        ys, xs = np.where(m)
        letters.append(dict(ch=ch, mask=m, x0=int(xs.min()), x1=int(xs.max()) + 1,
                            y0=int(ys.min()), y1=int(ys.max()) + 1, cost=float(c)))
    on = [L["y1"] for L in letters if L["ch"] in ON_BASE]
    # A letter's own bottom is its baseline (the printed guide line only for tails/marks that don't sit on it)
    baseline = (float(np.median(on)) if on else known_baseline if known_baseline is not None
                else float(np.median([L["y1"] for L in letters])))
    xs_ = [baseline - L["y0"] for L in letters if L["ch"] in XH]
    if xs_:
        xh = float(np.median(xs_))
    else:
        est = []
        for L in letters:
            r = (MEAN_DEFAULTS["asc_ratio"] if L["ch"] in ASC else MEAN_DEFAULTS["cap_ratio"] if (L["ch"] in CAPS or L["ch"] in "/()!?%&$")
                 else MEAN_DEFAULTS["digit_ratio"] if L["ch"] in DIGITS else None)
            if r:
                est.append((baseline - L["y0"]) / r)
        xh = float(np.median(est)) if est else 0.6 * est_h
    if fixed_xh is not None and not xs_ and ((not any(c.isalnum() for c in chars) and not any(c in "/()!?%&$" for c in chars))
                                              or all(c in DESC for c in chars)):
        xh = fixed_xh   # fill-in sheet: size caps/digits/marks against his real x-height on that sheet
    xh = max(xh, 4.0)
    return dict(slant=slant, alpha=alpha, ok=True, letters=letters, baseline=baseline, xh=xh,
                cost=float(np.mean([L["cost"] for L in letters])))


def stroke_width(mask):
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    v = dt[dt > 0]
    return float(2 * np.percentile(v, 75)) if v.size else 0.0


# ------------------------------------------------------------------ build
def build(min_conf):
    rejects = set(jload(DATA / "rejects.json", []))
    # versions the renderer kept having to swap out on the go get retired (never fill-in letters)
    rejects |= {k for k, v in (jload(DATA / "qa_flags.json", {}) or {}).items() if v >= 3 and not k.startswith("g")}
    lib = DATA / "library"
    if lib.exists():
        shutil.rmtree(lib)
    (lib / "glyphs").mkdir(parents=True)
    (lib / "words").mkdir(parents=True)
    glyphs, words, docs = {}, {}, []

    for lp in sorted((DATA / "labels").glob("*.json")):
        L = jload(lp)
        doc = L["doc"]
        meta = jload(DATA / "sources" / doc / "meta.json")
        if not meta:
            continue
        kind = meta.get("kind", "page")
        recs = {r["id"]: r for r in meta["words"]}
        order = {r["id"]: i for i, r in enumerate(meta["words"])}
        st = {k: [] for k in ("xh", "slant", "gap", "size", "resid", "stroke", "ink", "asc", "cap", "dig", "desc")}
        placed = []  # (line, x0, x1, baseline_page_y, xh) for line-level stats
        todo = [w for w, t in L["words"].items() if t and not t.startswith("#") and w in recs]
        sl = [deslant(np.asarray(Image.open(DATA / "sources" / doc / "words" / f"{w}.png").convert("L")))[1] for w in todo]
        hint = (float(np.median(sl)), 0.0 if kind == "gap" else 10.0) if sl else None
        sheet_xh = None
        if kind == "gap":
            xv = []
            for w_ in todo:
                if L["words"][w_] in XH:
                    r_ = analyse_word(np.asarray(Image.open(DATA / "sources" / doc / "words" / f"{w_}.png").convert("L")),
                                      L["words"][w_], meta.get("est_h", 40), recs[w_].get("baseline"), hint)
                    if r_ and r_.get("ok"):
                        xv.append(r_["xh"])
            sheet_xh = float(np.median(xv)) if xv else None
        n_glyph = 0
        for wid, text in L["words"].items():
            if not text or text.startswith("#") or wid not in recs:
                continue
            r = recs[wid]
            a = np.asarray(Image.open(DATA / "sources" / doc / "words" / f"{wid}.png").convert("L"))
            res = analyse_word(a, text, meta.get("est_h", 40), r.get("baseline"), hint, sheet_xh)
            if res is None or not res["ok"]:
                continue
            xh, base = res["xh"], res["baseline"]
            neat_tag = r.get("neat")
            if kind == "page":
                st["xh"].append(xh)
                st["slant"].append(res["slant"])
                placed.append((r["line"], r["bbox"][0], r["bbox"][2], r["bbox"][1] + base, xh, order[wid]))
            Ls = res["letters"]
            for A, B in zip(Ls[:-1], Ls[1:]):
                st["gap"].append((B["x0"] - A["x1"]) / xh)
            for Lt in Ls:
                if Lt["ch"] in XH:
                    st["size"].append((base - Lt["y0"]) / xh)
                if Lt["ch"] in ON_BASE and Lt["ch"] not in ".,":
                    st["resid"].append((Lt["y1"] - base) / xh)
                if Lt["ch"] in "bdhklt":
                    st["asc"].append((base - Lt["y0"]) / xh)
                if Lt["ch"] in CAPS:
                    st["cap"].append((base - Lt["y0"]) / xh)
                if Lt["ch"] in DIGITS:
                    st["dig"].append((base - Lt["y0"]) / xh)
                if Lt["ch"] in "gpqy":
                    st["desc"].append((Lt["y1"] - base) / xh)
            # glyphs
            for k, Lt in enumerate(Ls):
                if f"{doc}_{wid}_{k}" in rejects or (kind == "page" and (Lt["cost"] > min_conf or res["cost"] > min_conf * 0.8)):
                    continue
                m = cv2.dilate(Lt["mask"].astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
                y0, y1 = max(0, Lt["y0"] - 2), min(m.shape[0], Lt["y1"] + 2)
                x0, x1 = max(0, Lt["x0"] - 2), min(m.shape[1], Lt["x1"] + 2)
                g = np.where(m, res["alpha"], 0)[y0:y1, x0:x1].astype(np.uint8)
                gid = f"{doc}_{wid}_{k}"
                d = lib / "glyphs" / char_dir(Lt["ch"])
                d.mkdir(exist_ok=True)
                Image.fromarray(g, "L").save(d / f"{gid}.png")
                sw = stroke_width(Lt["mask"])
                st["stroke"].append(sw / xh)
                core = float(np.percentile(g[g > 30], 90)) if (g > 30).any() else 0
                st["ink"].append(core)
                glyphs.setdefault(Lt["ch"], []).append(dict(
                    id=gid, file=f"glyphs/{char_dir(Lt['ch'])}/{gid}.png", doc=doc,
                    top=round((base - y0) / xh, 3), bottom=round((y1 - base) / xh, 3),
                    w=round((x1 - x0) / xh, 3), src_xh=round(xh, 2), conf=round(1 - Lt["cost"], 3),
                    neat=neat_tag, stroke=round(sw / xh, 3), core=round(core, 1)))
                n_glyph += 1
            # whole word
            if " " not in text and res["cost"] < min_conf * 0.6 and len(Ls) >= 2:
                al = res["alpha"]
                ys, xs = np.where(al > 30)
                y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
                wid2 = f"{doc}_{wid}"
                Image.fromarray(al[y0:y1, x0:x1].astype(np.uint8), "L").save(lib / "words" / f"{wid2}.png")
                words.setdefault(text, []).append(dict(
                    id=wid2, file=f"words/{wid2}.png", doc=doc, label=text, top=round((base - y0) / xh, 3),
                    bottom=round((y1 - base) / xh, 3), w=round((x1 - x0) / xh, 3), src_xh=round(xh, 2),
                    neat=neat_tag))
        if kind != "page":
            continue
        # line-level stats
        word_base, slopes, wgaps = [], [], []
        by_line = {}
        for p in placed:
            by_line.setdefault(p[0], []).append(p)
        mx = float(np.median(st["xh"])) if st["xh"] else 1.0
        for ws in by_line.values():
            ws.sort(key=lambda p: p[1])
            for A, B in zip(ws[:-1], ws[1:]):
                if B[5] == A[5] + 1:  # only truly neighbouring words
                    wgaps.append((B[1] - A[2]) / mx)  # bbox pads cancel out on average
            if len(ws) >= 3:
                xs = np.array([(p[1] + p[2]) / 2 for p in ws]); ys = np.array([p[3] for p in ws])
                k, b = np.polyfit(xs, ys, 1)
                slopes.append(math.degrees(math.atan(k)))
                word_base += list((ys - (k * xs + b)) / mx)
        sd = lambda v: float(np.std(v)) if len(v) >= 2 else None  # noqa: E731
        stats = {
            "xh_cv": (float(np.std(st["xh"]) / np.mean(st["xh"])) if len(st["xh"]) >= 3 else None),
            "size_ratio_sd": sd(st["size"]), "base_resid_sd": sd(st["resid"]),
            "slant_sd": sd(st["slant"]) if len(st["slant"]) >= 3 else None,
            "letter_gap_sd": sd(st["gap"]), "word_gap_sd": sd(wgaps) if len(wgaps) >= 3 else None,
            "word_base_sd": sd(word_base) if len(word_base) >= 3 else None,
            "line_slope_sd": sd(slopes) if len(slopes) >= 3 else None,
            "slant_mean": float(np.mean(st["slant"])) if st["slant"] else None,
            "letter_gap_mean": float(np.mean(st["gap"])) if st["gap"] else None,
            "word_gap_mean": float(np.mean(wgaps)) if wgaps else None,
            "line_slope_mean": float(np.mean(slopes)) if slopes else None,
            "asc_ratio": float(np.median(st["asc"])) if st["asc"] else None,
            "cap_ratio": float(np.median(st["cap"])) if st["cap"] else None,
            "digit_ratio": float(np.median(st["dig"])) if st["dig"] else None,
            "desc_depth": float(np.median(st["desc"])) if st["desc"] else None,
            "stroke": float(np.median(st["stroke"])) if st["stroke"] else None,
            "ink_core": float(np.median(st["ink"])) if st["ink"] else None,
            "xh_px": mx,
        }
        sl = [min(1, max(0, (stats[f] - a) / (b - a))) for f, (a, b) in VAR_REF.items() if stats[f] is not None]
        neat = L.get("neatness")
        if neat is None:
            neat = 1 - float(np.mean(sl)) if sl else 0.5
        docs.append(dict(doc=doc, name=L.get("name"), neatness=round(neat, 3), n_words=len(placed),
                         n_glyphs=n_glyph, stats={k: (round(v, 4) if isinstance(v, float) else v) for k, v in stats.items()}))

    # neatness tags for glyphs/words from page docs
    dn = {d["doc"]: d["neatness"] for d in docs}
    for lst in list(glyphs.values()) + list(words.values()):
        for g in lst:
            if g["neat"] is None:
                g["neat"] = dn.get(g["doc"], 0.5)

    # neat->sloppy model per variability feature
    model = {}
    for f, (a, b) in VAR_REF.items():
        pts = [(d["neatness"], d["stats"][f]) for d in docs if d["stats"].get(f) is not None]
        mid = (a + b) / 2
        if len(pts) >= 2 and np.ptp([p[0] for p in pts]) >= 0.15:
            k, c = np.polyfit([p[0] for p in pts], [p[1] for p in pts], 1)
            neat_v, sloppy_v = max(0.2 * a, c + k), max(0.2 * a, c)
        elif pts:
            n0 = float(np.mean([p[0] for p in pts])); v0 = float(np.mean([p[1] for p in pts]))
            shape = lambda n: 0.55 + 0.9 * (1 - n)  # noqa: E731
            neat_v, sloppy_v = v0 * shape(1) / shape(n0), v0 * shape(0) / shape(n0)
        else:
            neat_v, sloppy_v = mid * 0.6, mid * 1.4
        # Shrink toward a sensible prior until there is enough writing (~80 words) to trust the data,
        # and never let a noisy estimate leave a plausible range.
        nw = sum(d["n_words"] for d in docs if d["stats"].get(f) is not None)
        wgt = min(1.0, nw / 300.0)
        pn, ps = a + 0.15 * (b - a), a + 0.85 * (b - a)
        neat_v = min(1.4 * b, max(0.6 * a, wgt * neat_v + (1 - wgt) * pn))
        sloppy_v = min(1.4 * b, max(neat_v, wgt * sloppy_v + (1 - wgt) * ps))
        model[f] = dict(neat=round(float(neat_v), 4), sloppy=round(float(sloppy_v), 4), n_docs=len(pts),
                        trust=round(wgt, 2))
    means = {}
    clamp = {"slant_mean": (-20, 25), "letter_gap_mean": (0.03, 0.5), "word_gap_mean": (0.5, 2.0),
             "line_slope_mean": (-3, 3), "asc_ratio": (1.3, 2.4), "cap_ratio": (1.3, 2.3),
             "digit_ratio": (1.2, 2.2), "desc_depth": (0.3, 1.2), "stroke": (0.05, 0.3), "ink_core": (40, 255)}
    for f, dflt in MEAN_DEFAULTS.items():
        v = [(d["stats"][f], d["n_words"]) for d in docs if d["stats"].get(f) is not None]
        nw = sum(n for _, n in v)
        wgt = min(1.0, nw / 60.0)
        est = float(np.average([x for x, _ in v], weights=[max(1, n) for _, n in v])) if v else dflt
        lo, hi = clamp[f]
        means[f] = round(min(hi, max(lo, wgt * est + (1 - wgt) * dflt)), 4)
    # quality control + gap filling (real > derived > synthetic)
    from hw_synth import derive, drop_outliers, synthesize
    for lst in glyphs.values():
        for g in lst:
            g.setdefault("src", "real")
    auto_rej = drop_outliers(glyphs, lib)
    derived = derive(glyphs, means, WANTED, lib)
    synth, synth_font = synthesize(glyphs, means, lib, WANTED)
    print(f"auto-dropped {len(auto_rej)} odd cuts; derived: {''.join(derived) or '-'}; "
          f"synthesized ({synth_font}): {''.join(synth) or '-'}")
    nn = [d["neatness"] for d in docs]
    task = dict(mean=round(float(np.mean(nn)), 3) if nn else 0.55,
                sd=round(max(0.06, float(np.std(nn))) if len(nn) >= 2 else 0.12, 3))
    coverage = {c: sum(1 for g in glyphs.get(c, []) if g.get("src") == "real") for c in WANTED}
    prof = dict(version=1, docs=docs, model=model, means=means, task_neatness=task, glyphs=glyphs,
                words=words, coverage=coverage, missing=[c for c in WANTED if coverage[c] == 0], derived=derived, synthesized=synth, auto_rejected=auto_rej,
                thin=[c for c in WANTED if 0 < coverage[c] < 3])
    lay = jload(DATA / "layout.json", {})
    if lay:
        prof["layout"] = dict(pages=lay)
    jsave(DATA / "profile.json", prof)
    report(prof)
    print(f"docs={len(docs)}  glyphs={sum(len(v) for v in glyphs.values())} ({len(glyphs)} chars)  "
          f"words={sum(len(v) for v in words.values())}")
    print(f"missing ({len(prof['missing'])}): {''.join(prof['missing'])}")
    print(f"report: {DATA / 'report' / 'report.md'}")


# ------------------------------------------------------------------ report
def variant_sheet(prof, chars, path, index, xh=34, maxv=40):
    rows = [c for c in chars if prof["glyphs"].get(c)]
    if not rows:
        return None
    W, RH = 90 + maxv * (int(1.6 * xh) + 10), int(3.4 * xh)
    im = Image.new("L", (W, RH * len(rows) + 10), 255)
    dr = ImageDraw.Draw(im)
    try:
        f = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 22)
    except OSError:
        f = ImageFont.load_default()
    for r, c in enumerate(rows):
        y_base = r * RH + int(2.2 * xh)
        dr.text((12, y_base - 26), f"{c}  ({len(prof['glyphs'][c])})", fill=90, font=f)
        dr.line((90, y_base, W - 10, y_base), fill=225)
        x = 95
        index.setdefault(c, [])
        for vi, g in enumerate(prof["glyphs"][c][:maxv], 1):
            index[c].append(g["id"])
            gi = Image.open(DATA / "library" / g["file"])
            s = xh / g["src_xh"]
            gi = gi.resize((max(1, int(gi.width * s)), max(1, int(gi.height * s))), Image.LANCZOS)
            im.paste(0, (x, int(y_base - g["top"] * xh)), gi)
            dr.text((x, y_base + int(0.9 * xh)), str(vi), fill=150, font=f)
            x += max(gi.width, 22) + 10
    im.save(path)
    return path


def report(prof):
    d = DATA / "report"
    d.mkdir(parents=True, exist_ok=True)
    index = {}
    sheets = [variant_sheet(prof, "abcdefghijklmnopqrstuvwxyz", d / "glyphs_lower.png", index),
              variant_sheet(prof, "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", d / "glyphs_upper_digits.png", index),
              variant_sheet(prof, ".,?!'-:;()$%/&+=\"", d / "glyphs_punct.png", index)]
    jsave(d / "variant_index.json", index)
    m, mo = prof["means"], prof["model"]
    lines = ["# Handwriting analysis", "",
             f"Documents: {len(prof['docs'])} · glyphs: {sum(len(v) for v in prof['glyphs'].values())} · "
             f"whole words: {sum(len(v) for v in prof['words'].values())}", "",
             "## Per document", "", "| doc | file | neatness | words | glyphs | slant ° | x-height px |", "|---|---|---|---|---|---|---|"]
    for doc in prof["docs"]:
        s = doc["stats"]
        lines.append(f"| {doc['doc']} | {doc['name']} | {doc['neatness']:.2f} | {doc['n_words']} | {doc['n_glyphs']} | "
                     f"{s['slant_mean'] if s['slant_mean'] is None else round(s['slant_mean'], 1)} | {round(s['xh_px'])} |")
    lines += ["", "## Your habits", "",
              f"- Slant: {m['slant_mean']:+.1f}° (positive = leans right)",
              f"- Letter spacing: {m['letter_gap_mean']:.2f} x-heights · word spacing: {m['word_gap_mean']:.2f} x-heights",
              f"- Tall letters reach {m['asc_ratio']:.2f}× x-height, capitals {m['cap_ratio']:.2f}×, tails drop {m['desc_depth']:.2f}×",
              f"- Pencil stroke ≈ {m['stroke']:.2f} x-heights thick",
              f"- Task neatness across your work: mean {prof['task_neatness']['mean']:.2f}, spread {prof['task_neatness']['sd']:.2f}",
              "", "## Neat vs sloppy model (value when neat → when sloppy)", "",
              "| feature | neat | sloppy | docs |", "|---|---|---|---|"]
    for f, v in mo.items():
        lines.append(f"| {f} | {v['neat']} | {v['sloppy']} | {v['n_docs']} (trust {v['trust']}) |")
    lines += ["", f"## Coverage", "", f"Not in your real writing yet: `{''.join(prof['missing']) or 'none'}`",
              f"Borrowed from your lowercase: `{''.join(prof.get('derived', [])) or 'none'}` · "
              f"style-matched stand-ins: `{''.join(prof.get('synthesized', [])) or 'none'}` · "
              f"odd cuts auto-dropped: {len(prof.get('auto_rejected', []))}",
              f"Fewer than 3 versions: `{''.join(prof['thin']) or 'none'}`", ""]
    for s in sheets:
        if s:
            lines.append(f"![{s.stem}]({s.name})")
    (d / "report.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-conf", type=float, default=0.55,
                    help="max letter-split cost to keep a glyph (lower = stricter)")
    ap.add_argument("--reject", default="",
                    help='drop badly cut versions by their number on the variant sheets, e.g. "a:1,3 e:2 Q:1"')
    a = ap.parse_args()
    if a.reject:
        idx = jload(DATA / "report" / "variant_index.json", {})
        rej = set(jload(DATA / "rejects.json", []))
        for part in a.reject.split():
            ch, nums = part.rsplit(":", 1)
            for n_ in nums.split(","):
                if ch in idx and 0 < int(n_) <= len(idx[ch]):
                    rej.add(idx[ch][int(n_) - 1])
        jsave(DATA / "rejects.json", sorted(rej))
        print(f"rejected {len(rej)} glyph(s) total")
    build(a.min_conf)


if __name__ == "__main__":
    main()
