"""Quality control + gap filling for the glyph library (called by hw_build.py).

drop_outliers  - removes letter cuts that don't look like the person's other versions of that letter
derive         - capitals/digits that are just bigger versions of his lowercase (C from c, 0 from o, I from l)
synthesize     - anything still missing: drawn from the bundled handwriting font closest to his
                 real letters, then matched to his letter heights, stroke width and pencil grain
Real > derived > synthetic: the renderer only falls back a tier when the better one is empty.
"""
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from hwcore import ASC, ASSETS, CAPS, DESC, DIGITS, char_dir

SAME_SHAPE = {c.upper(): c for c in "cosuvwxzkpy"}
SAME_SHAPE.update({"I": "l", "1": "l", "0": "o", "|": "l"})


def _vec(img, n=24):
    ys, xs = np.where(img > 30)
    if not len(ys):
        return None
    c = cv2.resize(img[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32), (n, n),
                   interpolation=cv2.INTER_AREA)
    c -= c.mean()
    s = c.std()
    return (c / s).ravel() if s > 0 else None


def drop_confusions(glyphs, lib, margin=0.08):
    """Drop real cuts that look more like a DIFFERENT letter's typical shape than their own
    (catches labels that slipped by one letter during splitting)."""
    vecs, means = {}, {}
    for ch, lst in glyphs.items():
        vs = {}
        for g in lst:
            if g.get("src", "real") != "real":
                continue
            v = _vec(np.asarray(Image.open(lib / g["file"]).convert("L")))
            if v is not None:
                vs[g["id"]] = v
        vecs[ch] = vs
        if len(vs) >= 3:
            means[ch] = np.mean(list(vs.values()), axis=0)
    dropped = []
    n = None
    for ch, vs in vecs.items():
        if ch not in means or len(vs) < 3:
            continue
        for gid, v in vs.items():
            n = n or len(v)
            if gid.startswith("g"):
                continue  # fill-in sheet letters have certain labels: they are the reference, never dropped
            own = (means[ch] * len(vs) - v) / (len(vs) - 1)  # own template without this cut
            c_own = float(v @ own) / (n * max(1e-6, own.std()))
            best = max(((float(v @ m) / (n * max(1e-6, m.std())), o) for o, m in means.items()
                        if o != ch and o.lower() != ch.lower()), default=(-9, None))
            if best[0] > c_own + margin:
                glyphs[ch] = [g for g in glyphs[ch] if g["id"] != gid]
                dropped.append(gid)
    return dropped


def drop_mis_sized(glyphs):
    """Cuts whose word's letter height was mis-measured render too big/small: drop them (not fill-in ones)."""
    out = []
    for ch, lst in glyphs.items():
        real = [g for g in lst if g.get("src", "real") == "real"]
        if len(real) < 3:
            continue
        med = float(np.median([g["src_xh"] * (g["top"] + g["bottom"]) for g in real]))  # ink height in px
        sx = float(np.median([g["src_xh"] for g in real]))
        ref = [g for g in real if g["id"].startswith("g")] or real      # fill-in letters are the reference
        mt = float(np.median([g["top"] for g in ref]))
        mw = float(np.median([g["w"] for g in real]))
        for g in real:
            if g["id"].startswith("g"):
                if g["w"] > 1.55 * mw and len(real) >= 4:              # fill-in letter with a box-border bar stuck on
                    lst.remove(g)
                    out.append(g["id"])
                continue
            partial = abs(g["top"] - mt) > 0.35 * max(0.3, mt)            # half a letter, or two stacked
            if partial or not (0.55 * sx <= g["src_xh"] <= 1.8 * sx):
                lst.remove(g)
                out.append(g["id"])
    return out


def drop_outliers(glyphs, lib):
    """Drop real cuts whose shape/aspect disagree strongly with the other versions."""
    dropped = drop_confusions(glyphs, lib) + drop_mis_sized(glyphs)
    for ch, lst in glyphs.items():
        real = [g for g in lst if g.get("src", "real") == "real"]
        if len(real) < 4:
            continue
        vecs = [_vec(np.asarray(Image.open(lib / g["file"]).convert("L"))) for g in real]
        idx = [i for i, v in enumerate(vecs) if v is not None]
        if len(idx) < 4:
            continue
        V = np.stack([vecs[i] for i in idx]) / len(vecs[idx[0]])
        C = V @ V.T
        asp = np.log([max(0.05, real[i]["w"]) / max(0.05, real[i]["top"] + real[i]["bottom"]) for i in idx])
        d = np.array([np.median(np.delete(1 - C[k], k)) + 0.5 * abs(asp[k] - np.median(asp))
                      for k in range(len(idx))])
        med = np.median(d)
        mad = np.median(np.abs(d - med)) + 1e-6
        for k in range(len(idx)):
            if d[k] > med + 3 * mad and d[k] > 0.45 and not real[idx[k]]["id"].startswith("g"):
                lst.remove(real[idx[k]])
                dropped.append(real[idx[k]]["id"])
    return dropped


def _target_top(ch, means):
    if ch in CAPS:
        return means["cap_ratio"]
    if ch in DIGITS:
        return means["digit_ratio"]
    if ch in ASC:
        return means["asc_ratio"]
    return None


def derive(glyphs, means, wanted, lib=None):
    made = []
    for tgt, src in SAME_SHAPE.items():
        if tgt not in wanted or any(g.get("src", "real") == "real" for g in glyphs.get(tgt, [])):
            continue
        srcs = [g for g in glyphs.get(src, []) if g.get("src", "real") == "real"]
        if not srcs:
            continue
        th = _target_top(tgt, means) or 1.6
        out = []
        for g in srcs[:8]:
            k = g["top"] / th                      # new src_xh = old * k  (scales the letter up)
            out.append(dict(g, id=f"der_{char_dir(tgt)}_{g['id']}", src="derived", conf=0.6,
                            src_xh=round(g["src_xh"] * k, 2), top=round(th, 3),
                            bottom=round(g["bottom"] / k, 3), w=round(g["w"] / k, 3)))
        if tgt == "I":
            for r in out:
                r["neat"] = 0.3                     # the plain bar is his rushed capital I
        glyphs[tgt] = [g for g in glyphs.get(tgt, []) if g.get("src", "real") == "real"] + out
        made.append(tgt)
    if lib is not None and "I" in wanted and not any(g.get("src", "real") == "real" for g in glyphs.get("I", [])):
        glyphs["I"] = glyphs.get("I", []) + serif_I(glyphs, means, lib)
    return made


def serif_I(glyphs, means, lib, n=8, seed=5):
    """His NEAT capital I has a bar at the top and bottom. Build it from his own strokes: one of his
    real l-stems plus two of his real dashes as the bars, each placed a little differently."""
    import random
    rng = random.Random(seed)
    stems = [g for g in glyphs.get("l", []) if g.get("src", "real") == "real"]
    bars = [g for g in glyphs.get("-", []) if g.get("src", "real") == "real"]
    if not stems or not bars:
        return []
    XHP = 40.0
    cap = means["cap_ratio"] * XHP
    d = lib / "glyphs" / char_dir("I")
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for k in range(n):
        st, b1, b2 = rng.choice(stems), rng.choice(bars), rng.choice(bars)
        sa = np.asarray(Image.open(lib / st["file"]).convert("L"), np.float32)
        ys = np.where(sa.max(1) > 30)[0]
        sa = sa[ys[0]:ys[-1] + 1]
        sh = int(round(cap * rng.uniform(0.95, 1.05)))
        sw = max(2, int(round(sa.shape[1] * sh / sa.shape[0])))
        stem = cv2.resize(sa, (min(sw, int(0.35 * XHP)), sh), interpolation=cv2.INTER_AREA)
        bw = int(round(XHP * rng.uniform(0.5, 0.78)))
        bimg = []
        for b in (b1, b2):
            ba = np.asarray(Image.open(lib / b["file"]).convert("L"), np.float32)
            xs = np.where(ba.max(0) > 30)[0]
            ys2 = np.where(ba.max(1) > 30)[0]
            ba = ba[ys2[0]:ys2[-1] + 1, xs[0]:xs[-1] + 1]
            bh = max(2, min(int(0.18 * XHP), int(round(ba.shape[0] * bw / max(1, ba.shape[1])))))
            bimg.append(cv2.resize(ba, (bw, bh), interpolation=cv2.INTER_AREA))
        pad = 6
        H, W = sh + 2 * pad + 4, bw + 2 * pad + 6
        c = np.zeros((H, W), np.float32)
        cx = W // 2 + rng.randint(-2, 2)
        c[pad:pad + sh, cx - stem.shape[1] // 2:cx - stem.shape[1] // 2 + stem.shape[1]] = stem
        for bi, yy in ((bimg[0], pad - bimg[0].shape[0] // 2 + rng.randint(0, 2)),
                       (bimg[1], pad + sh - bimg[1].shape[0] // 2 - rng.randint(0, 2))):
            x0 = cx - bi.shape[1] // 2 + rng.randint(-3, 3)
            yy = max(0, min(H - bi.shape[0], yy))
            x0 = max(0, min(W - bi.shape[1], x0))
            M = cv2.getRotationMatrix2D((bi.shape[1] / 2, bi.shape[0] / 2), rng.gauss(0, 2.5), 1)
            bi = cv2.warpAffine(bi, M, (bi.shape[1], bi.shape[0]))
            np.maximum(c[yy:yy + bi.shape[0], x0:x0 + bi.shape[1]], bi, out=c[yy:yy + bi.shape[0], x0:x0 + bi.shape[1]])
        base = pad + sh
        ys, xs = np.where(c > 30)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        g = c[y0:y1, x0:x1].clip(0, 255).astype(np.uint8)
        gid = f"der_I_serif_{k}"
        Image.fromarray(g, "L").save(d / f"{gid}.png")
        out.append(dict(id=gid, file=f"glyphs/{char_dir('I')}/{gid}.png", doc="derived", src="derived", conf=0.8,
                        neat=0.85, top=round((base - y0) / XHP, 3), bottom=round((y1 - base) / XHP, 3),
                        w=round((x1 - x0) / XHP, 3), src_xh=XHP))
    return out


def _font_x(fp, ch, xh_px):
    f100 = ImageFont.truetype(str(fp), 100)
    xb = f100.getbbox("x", anchor="ls")
    size = max(8, int(round(xh_px * 100 / max(1, -xb[1]))))
    f = ImageFont.truetype(str(fp), size)
    x0, y0, x1, y1 = f.getbbox(ch, anchor="ls")
    pad = 6
    im = Image.new("L", (x1 - x0 + 2 * pad, y1 - y0 + 2 * pad), 0)
    ImageDraw.Draw(im).text((pad - x0, pad - y0), ch, font=f, fill=255, anchor="ls")
    return np.asarray(im, dtype=np.float32), pad - y0  # image, baseline row


def pick_font(glyphs, lib):
    fonts = sorted((ASSETS / "fonts").glob("*.ttf"))
    chars = [c for c, lst in glyphs.items() if c.isalpha() and any(g.get("src", "real") == "real" for g in lst)][:24]
    if not chars:
        return fonts[0] if fonts else None
    best = (-9, fonts[0])
    for fp in fonts:
        sc = []
        for c in chars:
            fv = _vec(_font_x(fp, c, 40)[0])
            rv = [_vec(np.asarray(Image.open(lib / g["file"]).convert("L")))
                  for g in glyphs[c] if g.get("src", "real") == "real"][:6]
            rv = [v for v in rv if v is not None]
            if fv is not None and rv:
                sc.append(max(float(fv @ v) / len(fv) for v in rv))
        if sc and np.mean(sc) > best[0]:
            best = (float(np.mean(sc)), fp)
    return best[1]


def _stroke_px(a):
    m = (a > 100).astype(np.uint8)
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    v = dt[dt > 0]
    return float(2 * np.percentile(v, 75)) if v.size else 1.0


def synthesize(glyphs, means, lib, wanted, n_var=3, seed=11):
    have = {c for c, lst in glyphs.items() if lst}
    todo = [c for c in wanted if c not in have]
    if not todo:
        return [], None
    fp = pick_font(glyphs, lib)
    rng = random.Random(seed)
    XHP = 48
    made = []
    for ch in todo:
        out = []
        for k in range(n_var):
            a, base = _font_x(fp, ch, XHP)
            # match stroke width
            want = 0.7 * means["stroke"] * XHP  # distance-transform width overestimates real strokes
            have_w = _stroke_px(a)
            r = int(round((want - have_w) / 2))
            if r:
                ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abs(r) + 1, 2 * abs(r) + 1))
                a = cv2.dilate(a, ker) if r > 0 else cv2.erode(a, ker)
            ys = np.where(a.max(1) > 30)[0]
            if not len(ys):
                continue
            top_px, bot_px = base - ys[0], ys[-1] - base
            # match his letter heights
            tt = _target_top(ch, means)
            sy = (tt * XHP / max(1, top_px)) if tt else 1.0
            sx = sy * rng.uniform(0.92, 1.08)
            if ch in DESC and bot_px > 0:
                pass
            a = cv2.resize(a, (max(1, int(a.shape[1] * sx)), max(1, int(a.shape[0] * sy))), interpolation=cv2.INTER_AREA)
            base = base * sy
            # small random warp + pencil grain so variants differ
            h, w = a.shape
            M = cv2.getRotationMatrix2D((w / 2, base), rng.gauss(0, 2.5), 1)
            a = cv2.warpAffine(a, M, (w, h))
            grain = cv2.GaussianBlur(np.random.default_rng(rng.randrange(1 << 30)).uniform(0.7, 1.0, a.shape).astype(np.float32), (0, 0), 0.7)
            a = np.clip(a * grain * 0.85, 0, 255)
            ys, xs = np.where(a > 30)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            g = a[y0:y1, x0:x1].astype(np.uint8)
            gid = f"syn_{char_dir(ch)}_{k}"
            d = lib / "glyphs" / char_dir(ch)
            d.mkdir(parents=True, exist_ok=True)
            Image.fromarray(g, "L").save(d / f"{gid}.png")
            out.append(dict(id=gid, file=f"glyphs/{char_dir(ch)}/{gid}.png", doc="synth", src="synth", conf=0.3,
                            top=round((base - y0) / XHP, 3), bottom=round((y1 - base) / XHP, 3),
                            w=round((x1 - x0) / XHP, 3), src_xh=float(XHP), neat=0.5,
                            stroke=round(means["stroke"], 3), core=float(np.percentile(g[g > 30], 90))))
        if out:
            glyphs[ch] = out
            made.append(ch)
    return made, fp.name if fp else None
