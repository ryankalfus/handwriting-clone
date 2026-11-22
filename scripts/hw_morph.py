"""New letter shapes from the person's real versions, so exact copies aren't the only letters used.

Two real versions of a letter are put on a shared canvas (same x-height and baseline). Dense
optical flow measures how every stroke moves from one version to the other, which is how
that letter actually varies in his hand. A new letter warps both versions part-way along
that flow and blends them. t in (0, 1) gives an in-between shape; t slightly < 0 or > 1
extrapolates past his versions in the same directions his real variation goes. A final smooth
random deformation is scaled by how much this letter really varies (mean flow between versions),
so letters he writes inconsistently get more variety than letters he writes the same every time.
"""
import cv2
import numpy as np

XH = 40.0  # working x-height in pixels


def _prep(a, rec):
    s = XH / rec["src_xh"]
    img = cv2.resize(a, (max(1, int(round(a.shape[1] * s))), max(1, int(round(a.shape[0] * s)))),
                     interpolation=cv2.INTER_AREA)
    return img, rec["top"] * XH


def _pair(i1, t1, i2, t2):
    up = int(np.ceil(max(t1, t2))) + 6
    dn = int(np.ceil(max(i1.shape[0] - t1, i2.shape[0] - t2))) + 6
    W = max(i1.shape[1], i2.shape[1]) + 12
    out = []
    for a, t in ((i1, t1), (i2, t2)):
        c = np.zeros((up + dn, W), np.float32)
        y, x = max(0, int(round(up - t))), (W - a.shape[1]) // 2
        h, w = min(a.shape[0], c.shape[0] - y), min(a.shape[1], c.shape[1] - x)
        c[y:y + h, x:x + w] = a[:h, :w]
        out.append(c)
    return out[0], out[1], up


def _flow(c1, c2):
    b1 = np.clip(cv2.GaussianBlur(c1, (0, 0), 2.2) * 255, 0, 255).astype(np.uint8)
    b2 = np.clip(cv2.GaussianBlur(c2, (0, 0), 2.2) * 255, 0, 255).astype(np.uint8)
    return cv2.calcOpticalFlowFarneback(b1, b2, None, 0.5, 3, 13, 6, 5, 1.2, 0)


class Morpher:
    def __init__(self):
        self.cache = {}

    def pair(self, rec1, a1, rec2, a2):
        key = (rec1["id"], rec2["id"])
        if key not in self.cache:
            i1, t1 = _prep(a1, rec1)
            i2, t2 = _prep(a2, rec2)
            c1, c2, up = _pair(i1, t1, i2, t2)
            f12, f21 = _flow(c1, c2), _flow(c2, c1)
            m = c1 > 0.2
            var = float(np.mean(np.linalg.norm(f12, axis=2)[m])) / XH if m.any() else 0.05
            # do the strokes actually line up once c1 is warped onto c2? If not, blending would ghost.
            H, W = c1.shape
            gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
            w1 = cv2.remap(c1, gx - f12[..., 0], gy - f12[..., 1], cv2.INTER_LINEAR)
            b1, b2 = cv2.GaussianBlur(w1, (0, 0), 1.5), cv2.GaussianBlur(c2, (0, 0), 1.5)
            fit = float(np.abs(b1 - b2).sum() / max(1e-6, b2.sum()))
            self.cache[key] = (c1, c2, up, f12, f21, var, fit)
        return self.cache[key]

    def make(self, rec1, a1, rec2, a2, t, rng, jitter):
        """Returns (glyph image 0..1, metrics rec, letter variability)."""
        c1, c2, up, f12, f21, var, fit = self.pair(rec1, a1, rec2, a2)
        H, W = c1.shape
        gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        lo_top, hi_top = min(rec1["top"], rec2["top"]), max(rec1["top"], rec2["top"])
        if fit > 0.55:
            # versions too different to line up: bend ONE real version into a new shape instead
            out, jitter, var = c1.copy(), jitter * 1.5, 0.06
        else:
            for tt in (t, min(1.0, max(0.0, t))):     # second try: no extrapolation
                w1 = cv2.remap(c1, gx - tt * f12[..., 0], gy - tt * f12[..., 1], cv2.INTER_LINEAR)
                w2 = cv2.remap(c2, gx - (1 - tt) * f21[..., 0], gy - (1 - tt) * f21[..., 1], cv2.INTER_LINEAR)
                k = min(1.0, max(0.0, tt))
                out = (1 - k) * w1 + k * w2
                ys = np.where(out.max(1) > 0.3)[0]
                top = (up - ys[0]) / XH if len(ys) else lo_top
                if 0.9 * lo_top <= top <= 1.1 * hi_top:  # never taller/shorter than his real versions
                    break
        amp = jitter * max(0.03, min(0.25, var)) * XH
        if amp > 0.3:
            nr = np.random.default_rng(rng.randrange(1 << 30))
            dx = cv2.GaussianBlur(nr.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 0.35 * XH)
            dy = cv2.GaussianBlur(nr.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 0.35 * XH)
            n = max(1e-6, float(np.abs(dx).max()), float(np.abs(dy).max()))
            out = cv2.remap(out, gx + dx / n * amp, gy + dy / n * amp, cv2.INTER_LINEAR)
        on = out > 0.1
        core = float(np.percentile(out[on], 90)) if on.any() else 1.0
        out = np.clip(out / max(core, 1e-3), 0, 1)          # keep the one even pencil tone
        if fit <= 0.55 and (out > 0.4).sum() < 0.8 * min((c1 > 0.4).sum(), (c2 > 0.4).sum()):
            return None                                      # blending washed out a stroke (e.g. a B's stem)
        ys, xs = np.where(out > 0.05)
        if not len(ys):
            return None
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        rec = dict(id=f"m:{rec1['id']}:{rec2['id']}:{t:.2f}", src_xh=XH, top=float(up - y0) / XH,
                   bottom=float(y1 - up) / XH, w=float(x1 - x0) / XH, conf=1.0,
                   neat=(rec1.get("neat", 0.5) + rec2.get("neat", 0.5)) / 2)
        return out[y0:y1, x0:x1], rec, var
