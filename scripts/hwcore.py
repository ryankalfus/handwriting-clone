"""Shared helpers for the handwriting-clone skill (paths, IO, photo cleanup)."""
import json
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

SKILL = Path(__file__).resolve().parent.parent
# This install's own handwriting data, and nothing else. Deliberately NOT overridable (no env var / flag),
# so the skill can never be pointed at another person's samples or another skill's library.
DATA = SKILL / "data"
ASSETS = SKILL / "assets"

# Character classes (used for baseline / x-height estimation)
XH = set("acemnorsuvwxz")            # x-height letters, sit on baseline
ASC = set("bdfhklt")                  # ascenders
DESC = set("gjpqy")                   # descenders
CAPS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DIGITS = set("0123456789")
ON_BASE = XH | ASC | CAPS | DIGITS | set("i!?.%&#")

# Rough relative widths of characters (in x-height units) for letter splitting.
REL_W = {c: 1.0 for c in "abcdeghknopqsuvxyz"}
REL_W.update({"m": 1.55, "w": 1.45, "i": 0.38, "l": 0.38, "j": 0.45, "f": 0.6, "t": 0.62,
              "r": 0.7, "1": 0.5, "I": 0.45, "M": 1.6, "W": 1.6, ".": 0.25, ",": 0.25,
              "'": 0.25, ":": 0.25, ";": 0.3, "!": 0.3, "-": 0.6, "$": 0.9, "%": 1.3,
              "&": 1.1, "(": 0.45, ")": 0.45, "/": 0.7, "=": 0.9, "+": 0.9, "?": 0.8})
for _c in CAPS:
    REL_W.setdefault(_c, 1.2)
for _c in DIGITS:
    REL_W.setdefault(_c, 0.95)


def rel_w(c):
    return REL_W.get(c, 0.9)


def jload(p, default=None):
    p = Path(p)
    if not p.exists():
        return default
    with open(p) as f:
        return json.load(f)


def jsave(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    tmp.replace(p)


def char_dir(c):
    """Filesystem-safe folder name for a character."""
    return f"u{ord(c):04x}"


# ------------------------------------------------------------------ photos
def load_photo(path, max_side=3400):
    """Load a photo (jpg/png/heic/webp) as BGR uint8, EXIF-rotated, downscaled."""
    path = Path(path)
    src = path
    tmpdir = None
    if path.suffix.lower() in (".heic", ".heif"):
        tmpdir = tempfile.mkdtemp()
        src = Path(tmpdir) / (path.stem + ".jpg")
        subprocess.run(["sips", "-s", "format", "jpeg", str(path), "--out", str(src)],
                       check=True, capture_output=True)
    im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    s = max_side / max(im.size)
    if s < 1:
        im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def _order(pts):
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
                    dtype=np.float32)


def find_page(bgr):
    """Return the 4 corners of the sheet of paper if it is clearly visible, else None."""
    h, w = bgr.shape[:2]
    sc = 1000 / max(h, w)
    small = cv2.resize(bgr, (int(w * sc), int(h * sc)))
    g = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    best = None
    for lo, hi in ((30, 90), (60, 160)):
        e = cv2.dilate(cv2.Canny(g, lo, hi), np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(e, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:5]:
            a = cv2.contourArea(c)
            if a < 0.3 * small.shape[0] * small.shape[1]:
                break
            ap = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
            if len(ap) == 4 and (best is None or a > best[0]):
                best = (a, ap.reshape(4, 2) / sc)
    if best is None:
        return None
    # A quad touching the image border on all sides is just the photo frame.
    q = best[1]
    if (q[:, 0].min() < 5 and q[:, 1].min() < 5 and q[:, 0].max() > w - 6 and q[:, 1].max() > h - 6):
        return None
    return _order(q)


def warp_page(bgr, quad):
    tl, tr, br, bl = quad
    W = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    H = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    M = cv2.getPerspectiveTransform(quad, np.float32([[0, 0], [W, 0], [W, H], [0, H]]))
    return cv2.warpPerspective(bgr, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def flatten_light(gray):
    """Divide out uneven lighting/shadows -> paper ~255, ink keeps its relative darkness."""
    h, w = gray.shape
    k = max(15, (max(h, w) // 60) | 1)
    bg = cv2.dilate(gray, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.GaussianBlur(bg, (0, 0), k / 2)
    norm = np.clip(gray.astype(np.float32) / np.maximum(bg, 1) * 255.0, 0, 255)
    return norm.astype(np.uint8)


def components(mask, min_area=1):
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a >= min_area:
            out.append(dict(i=i, x0=int(x), y0=int(y), x1=int(x + w), y1=int(y + h), area=int(a)))
    return out, lab


def two_means_threshold(vals, lo=None, hi=None):
    """Split 1-D values into two clusters; return midpoint threshold (or None)."""
    v = np.sort(np.asarray(vals, dtype=np.float64))
    if len(v) < 4 or v[-1] - v[0] < 1e-6:
        return None
    best, bt = None, None
    for k in range(1, len(v)):
        a, b = v[:k], v[k:]
        cost = ((a - a.mean()) ** 2).sum() + ((b - b.mean()) ** 2).sum()
        if best is None or cost < best:
            best, bt = cost, (a.mean() + b.mean()) / 2
    if lo is not None:
        bt = max(lo, bt)
    if hi is not None:
        bt = min(hi, bt)
    return bt


def to_alpha_png(dark, mask, path):
    """Save ink darkness (0..255) restricted to mask as an L-mode PNG (white=ink)."""
    a = np.where(mask, dark, 0).astype(np.uint8)
    Image.fromarray(a, "L").save(path)
