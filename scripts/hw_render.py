"""Render text in the current person's handwriting using the library built by hw_build.py.

Randomness is hierarchical ("variance of the variance"):
  task   -> neatness drawn from their real task-to-task spread, per-feature variance
            multipliers, a mood (size/slant shift), how much neatness wanders, fatigue
  word   -> its own neatness around the task's, its own variance multiplier,
            size / slant / baseline offset, maybe a whole real word image
  letter -> one of the person's real versions of that letter (picked by neatness, no
            immediate repeats), jittered size / rotation / baseline / spacing and a small
            elastic warp so reused versions never look identical

usage:
  hw_render.py text "words to write" --out out.png [--width-in 6.5] [--xh-pt 7] [--neatness 0.7] [--seed N]
  hw_render.py spec spec.json          # fill regions on a PDF / blank pages -> print-safe PDF
  hw_render.py lines file.pdf          # list printed text lines + positions (to plan regions)
"""
import argparse
import math
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hwcore import ASC, ASSETS, CAPS, DATA, DESC, DIGITS, XH, jload, jsave  # noqa: E402

FALLBACK_FONT = ASSETS / "fonts" / "ReenieBeanie.ttf"
FEAT_MAX = {"xh_cv": 0.2, "size_ratio_sd": 0.18, "base_resid_sd": 0.15, "slant_sd": 7.0, "letter_gap_sd": 0.25,
            "word_gap_sd": 0.45, "word_base_sd": 0.22, "line_slope_sd": 2.0}
VAR = ["xh_cv", "size_ratio_sd", "base_resid_sd", "slant_sd", "letter_gap_sd", "word_gap_sd",
       "word_base_sd", "line_slope_sd"]


def lognorm(rng, sigma):
    return math.exp(rng.gauss(0, sigma))


# ------------------------------------------------------------------ randomness hierarchy
def clip(v, lo, hi):
    return min(hi, max(lo, v))


QS = [0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95]


def qsample(rng, qs, lo, hi):
    """Draw from an empirical distribution stored as quantiles (hw_layout.QS)."""
    xs = [0.0] + QS + [1.0]
    ys = [qs[0] - (qs[1] - qs[0])] + list(qs) + [qs[-1] + (qs[-1] - qs[-2])]
    return clip(float(np.interp(rng.random(), xs, ys)), lo, hi)


class Personality:
    """A page style. Usually GENERATED: a draw from a statistical model of how his pages differ
    (mean + shrunk correlations, spread inflated past what his pages show, plus a push along his
    neat<->sloppy direction). Sometimes it stays near one of his real pages (or a blend of two).
    How often it stays near a real page, how far it is pushed, and how much it is jittered are
    all random per task: the randomness of the randomness."""
    SIGNED = ("slope_mean", "slant_mean", "fatigue", "wobble_phi", "size_phi")
    KEYS = ("slope_sd", "curv_sd", "wobble_mean", "wobble_cv", "wobble_phi", "size_sd", "size_phi",
            "overlap_rate", "overlap_depth", "line_spacing_mean", "line_spacing_cv", "margin_sd",
            "slant_sd", "sloppy_line_share", "switch_rate", "fatigue", "slope_mean", "slant_mean")

    def __init__(self, pages, rng, force=None, all_pages=None):
        self.v, self.src = {}, "defaults"
        if not pages:
            return
        self.tau = rng.uniform(0.05, 0.4)
        hit = [p for p in (all_pages or pages) if force and str(force).lower() in str(p.get("name", "")).lower()]
        if hit:
            self.tau = 0.05
            self._near_real(pages, rng, hit[0])
        elif rng.random() < rng.uniform(0.05, 0.3):
            self._near_real(pages, rng)
        else:
            self._generated(pages, rng)
        src = rng.choice(pages)
        for k in ("word_gap_q", "letter_gap_q"):
            q = src.get(k)
            self.v[k] = [x * lognorm(rng, 0.15) for x in q] if q else None

    def _near_real(self, pages, rng, forced=None):
        A, B = (forced, forced) if forced else (rng.choice(pages), rng.choice(pages))
        w = rng.betavariate(2, 2) if A is not B else 1.0
        for k in self.KEYS:
            xa, xb = A.get(k), B.get(k)
            if xa is None and xb is None:
                continue
            x = xa if xb is None else xb if xa is None else w * xa + (1 - w) * xb
            self.v[k] = x + rng.gauss(0, self.tau * (abs(x) + 0.3)) if k in self.SIGNED else x * lognorm(rng, self.tau)
        self.src = f"near {A.get('name')}x{w:.2f}+{B.get('name')} tau={self.tau:.2f}"

    def _generated(self, pages, rng):
        keys = [k for k in self.KEYS if sum(p.get(k) is not None for p in pages) >= 3]
        X = np.array([[p.get(k) if p.get(k) is not None else np.nan for k in keys] for p in pages], float)
        pos = [k not in self.SIGNED for k in keys]
        for j, ps in enumerate(pos):
            if ps:
                X[:, j] = np.log(np.maximum(X[:, j], 1e-4))
        mu = np.nanmean(X, 0)
        sd = np.maximum(np.nanstd(X, 0), 0.08)
        Z = np.where(np.isnan(X), 0.0, (X - mu) / sd)
        C = np.corrcoef(Z.T) if len(pages) >= 3 else np.eye(len(keys))
        C = 0.4 * np.nan_to_num(C) + 0.6 * np.eye(len(keys))      # shrink: only a handful of pages
        L = np.linalg.cholesky(C + 1e-6 * np.eye(len(keys)))
        infl = lognorm(rng, 0.3) * 1.5                           # wider than his real pages
        z = L @ np.array([rng.gauss(0, 1) for _ in keys]) * infl
        if "wobble_mean" in keys:                                # push along his neat<->sloppy axis
            jw = keys.index("wobble_mean")
            axis = Z[np.argmax(Z[:, jw])] - Z[np.argmin(Z[:, jw])]
            axis /= max(1e-6, np.linalg.norm(axis))
            push = rng.gauss(0, 1.0)
            z = z + push * axis * np.sqrt(len(keys)) * 0.5
        else:
            push = 0.0
        x = mu + sd * np.clip(z, -3.5, 3.5)
        for j, k in enumerate(keys):
            self.v[k] = float(np.exp(x[j])) if pos[j] else float(x[j])
        self.src = f"generated infl={infl:.2f} push={push:+.2f}"

    def get(self, k, d):
        v = self.v.get(k)
        return d if v is None else v


STYLE_WORDS = {"very tight": 0.55, "tight": 0.72, "normal": 1.0, "loose": 1.35, "very loose": 1.6,
               "narrow": 0.9, "wide": 1.1, "small": 0.88, "big": 1.12, "consistent": 0.9,
               "inconsistent": 0.25, "low": 0.5, "high": 1.6, "none": 0.0}
STYLE_KEYS = ("tidy", "neatness", "chaos", "width", "letter_spacing", "word_spacing", "slant", "size", "consistency",
              "fatigue", "overlap", "new_shapes", "page", "wobble", "line_drift", "whole_words")


def parse_style(st):
    """'neatness=0.8 slant=6 letter_spacing=tight page=IMG_3187' or a dict -> dict."""
    if not st:
        return {}
    if isinstance(st, str):
        st = dict(tok.split("=", 1) for tok in st.replace(",", " ").split() if "=" in tok)
    out = {}
    for k, v in st.items():
        k = k.strip().lower().replace("-", "_")
        if k not in STYLE_KEYS:
            print(f"warning: unknown style key '{k}' (known: {', '.join(STYLE_KEYS)})")
            continue
        if k == "page":
            out[k] = str(v)
            continue
        v = str(v).strip().lower().replace("_", " ")
        out[k] = STYLE_WORDS[v] if v in STYLE_WORDS else float(v)
    return out


def load_style(override=None):
    """The person's always-on choices (data/style_defaults.json) with this request's choices on top."""
    base = jload(DATA / "style_defaults.json", {}) or {}
    st = parse_style(base)
    st.update(parse_style(override))
    return st


class Task:
    def __init__(self, prof, rng, neatness=None, style=None):
        self.p, self.rng = prof, rng
        style = dict(style or {})
        if style.get("neatness") is not None:
            neatness = style["neatness"]
        tn = prof.get("task_neatness", {"mean": 0.55, "sd": 0.12})
        if neatness is None:
            trust = float(np.mean([v.get("trust", 1.0) for v in prof["model"].values()]))
            m = min(0.92, max(0.08, trust * tn["mean"] + (1 - trust) * 0.55))
            s = max(0.05, tn["sd"])
            k = max(2.0, m * (1 - m) / s ** 2 - 1)
            neatness = rng.betavariate(m * k, (1 - m) * k)
        self.neat = float(neatness)
        sloppy = 1 - self.neat
        allp = list((prof.get("layout") or {}).get("pages", {}).values())
        pages = [v for v in allp if v.get("n_lines", 0) >= 8]
        P = self.pers = Personality(pages, rng, style.get("page"), allp)
        # his line spacing was measured in letter-body heights (~1.3 x-heights)
        self.line_gap_auto = clip(P.get("line_spacing_mean", 2.9) * 1.3, 3.2, 4.6)
        tau = 0.08 + 0.20 * sloppy
        self.vv = {f: lognorm(rng, tau) for f in VAR}
        self.size = lognorm(rng, 0.05)
        self.slant = prof["means"]["slant_mean"] + rng.gauss(0, 1.5)
        self.line_slope = clip(rng.gauss(prof["means"].get("line_slope_mean", 0.0), 0.3), -0.6, 0.6)
        # photo measurements include lens/page-curl distortion, so geometric stats are discounted
        self.slope_sd = clip(P.get("slope_sd", 0.8) * 0.35, 0.08, 0.7)
        self.curv_sd = clip(P.get("curv_sd", 0.15) * 0.4, 0.0, 0.2)
        self.spacing_cv = clip(P.get("line_spacing_cv", 0.06), 0.0, 0.18)
        self.margin_sd = clip(P.get("margin_sd", 0.3) * 0.4, 0.0, 0.6)
        self.word_neat_sd = abs(rng.gauss(0.05 + 0.08 * sloppy, 0.025))
        self.tau_word = 0.10 + 0.30 * sloppy
        # neat/sloppy stretches: two-state Markov chain, stationary share and stickiness from his pages
        share = clip(P.get("sloppy_line_share", 0.3) * (0.5 + sloppy), 0.02, 0.85)
        sw = clip(P.get("switch_rate", 0.4) * rng.uniform(0.25, 0.5), 0.02, 0.45)
        self.p01, self.p10 = clip(2 * sw * share, 0.01, 0.9), clip(2 * sw * (1 - share), 0.01, 0.9)
        self.state = 1 if rng.random() < share else 0
        self.drop = rng.uniform(0.12, 0.32)              # how much a sloppy stretch costs (random too)
        self.fatigue = clip(rng.gauss(0.04 + 0.1 * sloppy + 0.15 * max(0.0, P.get("fatigue", 0.0)), 0.05), -0.1, 0.35)
        # slowly wandering size and baseline (AR(1) along the text)
        self.size_phi = clip(P.get("size_phi", 0.3), -0.3, 0.85)
        self.size_sd = clip(P.get("size_sd", 0.08) * 0.6, 0.02, 0.12)
        self.base_phi = clip(P.get("wobble_phi", 0.2), -0.3, 0.85)
        self.base_sd = clip(P.get("wobble_mean", 0.12) * 0.8, 0.03, 0.22)
        self.prev_size = self.prev_base = 0.0
        # measured on blobs, so pieces of one letter (t-bars, i-dots) inflate it: discount
        self.overlap_rate = clip(P.get("overlap_rate", 0.2) * 0.45, 0.0, 0.25)
        self.overlap_depth = clip(P.get("overlap_depth", 0.08) * 0.6, 0.01, 0.12)
        self.recent = {}
        # more randomness -- and every amount below is itself random per task
        self.chaos = clip(lognorm(rng, 0.35), 0.6, 1.9) * (0.55 + 0.9 * sloppy)   # neat writing is steadier
        self.word_neat_sd *= self.chaos
        self.tau_word *= self.chaos
        self.size_sd = clip(self.size_sd * self.chaos, 0.02, 0.14)
        self.base_sd = clip(self.base_sd * self.chaos, 0.03, 0.24)
        self.wide = clip(lognorm(rng, 0.07), 0.86, 1.16)          # narrow vs wide writing today
        self.lgap_scale = clip(lognorm(rng, 0.2), 0.6, 1.6)      # tight vs loose letters
        self.wgap_scale = clip(lognorm(rng, 0.15), 0.7, 1.4)     # tight vs loose words
        self.p_big, self.p_small = (rng.uniform(0, 0.05) * 2 * sloppy, rng.uniform(0, 0.05) * 2 * sloppy)
        self.p_new = rng.uniform(0.65, 0.95)      # share of letters drawn as NEW shapes, not exact copies
        self.reach = rng.uniform(0.05, 0.3)       # how far new shapes may go past his real versions
        self.p_whole = rng.uniform(0.05, 0.25)    # whole-word reuse (always lightly warped)
        self.slant += rng.gauss(0, 2.0)
        self.apply_style(style)
        # PRECISION: neat writing is precise. Every source of variation shrinks together as neatness rises
        # (neat = on the lines, steady, nothing jumbled). Even everyday writing is somewhat tighter.
        n = self.neat
        self.prec = clip((n - 0.5) / 0.45, 0.0, 1.0) ** 1.2
        # EXTRA-NEAT ('tidy'): beyond neat, every letter is straightened to one slant and sized to its exact
        # class height. On request ('extra neat', 'best handwriting') or automatically at neatness 1.0.
        self.tidy = clip(self.style.get("tidy", (n - 0.96) / 0.04), 0.0, 1.0)
        # parts of 'tidy': neat writing evens out letter SIZE (so every word grows together) but keeps each
        # letter's natural tilt and spot; extra-neat also straightens tilt and snaps letters onto the line
        self.even_size = max(self.tidy, 0.8 * self.prec)
        self.damp = 0.85 * (1 - 0.88 * self.prec)
        d = self.damp
        mean_slant = prof["means"]["slant_mean"]
        if "slant" not in self.style:
            self.slant = mean_slant + (self.slant - mean_slant) * d
        for k in ("base_sd", "size_sd", "slope_sd", "curv_sd", "spacing_cv", "margin_sd", "word_neat_sd",
                  "overlap_rate", "p_big", "p_small", "reach", "fatigue"):
            setattr(self, k, getattr(self, k) * d)
        self.line_slope *= d
        self.drop *= d
        self.tau_word *= d
        self.chaos *= d
        self.size = 1 + (self.size - 1) * d
        for k in ("wide", "lgap_scale", "wgap_scale"):
            if k not in ("wide",) or "width" not in self.style:
                setattr(self, k, 1 + (getattr(self, k) - 1) * d)

    def apply_style(self, st):
        """Lock in whatever was chosen; everything not mentioned stays random."""
        self.style = st
        if "chaos" in st:
            r = max(0.1, st["chaos"]) / self.chaos
            self.chaos = st["chaos"]
            self.word_neat_sd *= r
            self.tau_word *= r
            self.size_sd = clip(self.size_sd * r, 0.01, 0.16)
            self.base_sd = clip(self.base_sd * r, 0.01, 0.26)
        if "width" in st:
            self.wide = clip(st["width"], 0.75, 1.3)
        if "letter_spacing" in st:
            self.lgap_scale = clip(st["letter_spacing"], 0.3, 2.0)
        if "word_spacing" in st:
            self.wgap_scale = clip(st["word_spacing"], 0.5, 1.8)
        if "slant" in st:
            self.slant = clip(st["slant"], -25, 30)
        if "size" in st:
            self.size = clip(st["size"], 0.6, 1.6)
        if "consistency" in st:                      # 1 = same all the way through, 0 = all over the place
            c = clip(st["consistency"], 0, 1)
            share, sw = 0.6 * (1 - c), 0.45 * (1 - c)
            self.p01, self.p10 = clip(2 * sw * share, 0.0, 0.9), clip(2 * sw * (1 - share), 0.01, 0.9)
            self.state = 0
            self.word_neat_sd *= 0.4 + 0.8 * (1 - c)
            self.drop *= (1 - c) + 0.1          # a 'consistent' task barely dips when it does switch
        if "fatigue" in st:
            self.fatigue = clip(st["fatigue"], -0.1, 0.5)
        if "overlap" in st:
            self.overlap_rate = clip(st["overlap"], 0, 0.5)
        if "new_shapes" in st:
            self.p_new = clip(st["new_shapes"], 0, 1)
        if "wobble" in st:
            self.base_sd = clip(self.base_sd * st["wobble"], 0.0, 0.3)
        if "line_drift" in st:
            self.slope_sd *= st["line_drift"]
            self.curv_sd *= st["line_drift"]
        if "whole_words" in st:
            self.p_whole = clip(st["whole_words"], 0, 1)

    def feat(self, f, n):
        m = self.p["model"][f]
        v = (m["sloppy"] + (m["neat"] - m["sloppy"]) * n) * self.vv[f]
        return min(FEAT_MAX[f] * (0.6 + 0.4 * (1 - n)), max(0.0, v)) * getattr(self, "damp", 1.0)

    def word(self, progress):
        r = self.rng
        if self.state == 0 and r.random() < self.p01:
            self.state = 1
        elif self.state == 1 and r.random() < self.p10:
            self.state = 0
        center = self.neat - self.fatigue * progress - self.state * self.drop
        return Word(self, clip(r.gauss(center, self.word_neat_sd), 0.0, 1.0))

    def word_gap(self, n):
        qs = self.pers.get("word_gap_q", None)
        g = qsample(self.rng, qs, 0.45, 2.4) if qs else self.p["means"]["word_gap_mean"] + self.rng.gauss(0, 0.3)
        return clip(g * (0.9 + 0.25 * (1 - n)) * self.wgap_scale, 0.45 + 0.4 * n, 2.4)   # neat = reliable gaps


class Word:
    def __init__(self, task, n):
        r = task.rng
        self.t, self.n = task, n
        self.V = min(1.8, lognorm(r, task.tau_word))      # this word's own variance multiplier
        k = 0.6 + 0.8 * (1 - n)                              # sloppier words swing more
        task.prev_size = task.size_phi * task.prev_size + r.gauss(0, task.size_sd * k * math.sqrt(1 - task.size_phi ** 2))
        task.prev_base = task.base_phi * task.prev_base + r.gauss(0, task.base_sd * k * math.sqrt(1 - task.base_phi ** 2))
        self.size = task.size * clip(math.exp(task.prev_size), 0.8, 1.22)
        u = r.random()
        if u < task.p_big:
            self.size *= r.uniform(1.12, 1.28)       # the odd word written bigger
        elif u < task.p_big + task.p_small:
            self.size *= r.uniform(0.78, 0.9)        # ...or cramped smaller
        self.base = clip(task.prev_base, -0.35, 0.35)
        self.slant = task.slant + clip(r.gauss(0, task.feat("slant_sd", n)), -9, 9)
        self.gap_mean = task.p["means"]["letter_gap_mean"]

    def letter(self):
        r, t, n, V = self.t.rng, self.t, self.n, self.V
        bsd = t.feat("base_resid_sd", n) * V
        if r.random() < t.overlap_rate * (0.7 + 0.6 * (1 - n)):
            gap = -abs(r.gauss(t.overlap_depth, 0.5 * t.overlap_depth))   # letters run into each other
        else:
            gap = self.gap_mean * t.lgap_scale + r.gauss(0, t.feat("letter_gap_sd", n) * 0.7 * V)
        return dict(scale=self.size * clip(1 + r.gauss(0, t.feat("size_ratio_sd", n) * 0.6 * V), 0.86, 1.15),
                    base=clip(r.gauss(0, bsd), -0.4, 0.4),  # word offset is applied once, in layout
                    rot=clip(r.gauss(0, 20 * bsd + 0.6 * t.damp), -9, 9),
                    gap=clip(gap, -0.3, 0.55),
                    warp=min(0.07, (0.015 + 0.05 * (1 - n)) * V * t.damp))


# ------------------------------------------------------------------ glyph sources
class Library:
    def __init__(self, prof):
        self.p = prof
        self.cache = {}
        self.fallback_used = set()
        self._qa_init()
        self._font = {}

    def img(self, rec):
        if rec["file"] not in self.cache:
            a = np.asarray(Image.open(DATA / "library" / rec["file"]).convert("L"), dtype=np.float32)
            core = np.percentile(a[a > 30], 90) if (a > 30).any() else 255
            # even tone: every glyph core -> 1, and lighter strokes inside a letter (a pale B stem) are lifted
            self.cache[rec["file"]] = np.clip(a / max(core, 1), 0, 1) ** 0.65
        return self.cache[rec["file"]]

    # ---- on-the-go checks + memory of versions that keep failing
    def _qa_init(self):
        if not hasattr(self, "flags"):
            self.flags, self.fixes = {}, []
            self.bad = {k for k, v in (jload(DATA / "qa_flags.json", {}) or {}).items() if v >= 2}

    def templates(self):
        if not hasattr(self, "_tpl"):
            from hw_synth import _vec
            self._tpl = {}
            for c, lst in self.p["glyphs"].items():
                ref = [g for g in lst if g["id"].startswith("g")] or [g for g in lst if g.get("src", "real") == "real"]
                vs = [_vec((self.img(g) * 255).astype(np.uint8)) for g in ref[:12]]
                vs = [v for v in vs if v is not None]
                if len(vs) >= 2:
                    self._tpl[c] = np.mean(vs, 0)
        return self._tpl

    def check(self, ch, a, rec, pool_n):
        """None if the letter looks right; otherwise a short reason."""
        if (a > 0.3).sum() < 12:
            return "faint/empty"
        if ch in "ij":
            lo, hi = 0.8, 2.0            # the dot sits above the x-height
        elif ch in XH or ch in DESC:
            lo, hi = 0.5, 1.45
        elif ch in ASC or ch in CAPS or ch in DIGITS:
            lo, hi = 1.1, 2.4
        else:
            lo = hi = None
        if lo is not None and not (lo <= rec.get("top", 1.0) <= hi):
            return f"wrong height ({rec.get('top', 0):.2f})"
        if ch.isalnum() and pool_n >= 3:
            from hw_synth import _vec
            tpl = self.templates()
            if ch in tpl:
                v = _vec((np.clip(a, 0, 1) * 255).astype(np.uint8))
                if v is None:
                    return "unreadable"
                n = len(v)
                own = float(v @ tpl[ch]) / (n * max(1e-6, tpl[ch].std()))
                ok = set(SIMILAR.get(ch, "")) | {ch.lower(), ch.upper()}
                best = max(((float(v @ m) / (n * max(1e-6, m.std())), o) for o, m in tpl.items()
                            if o not in ok and o.isalnum()),
                           default=(-9, None))
                if best[0] > own + 0.12:
                    return f"looks like '{best[1]}'"
        return None

    def remember(self, ch, a):
        from hw_synth import _vec
        self._prev = (ch, None if a is None else _vec((np.clip(a, 0, 1) * 255).astype(np.uint8)))

    def twin(self, ch, a):
        """True if this letter would look nearly identical to the same/look-alike letter right before it."""
        pc, pv = getattr(self, "_prev", (None, None))
        if pv is None or not (pc == ch or pc in SIMILAR.get(ch, "") or ch in SIMILAR.get(pc, "")):
            return False
        from hw_synth import _vec
        v = _vec((np.clip(a, 0, 1) * 255).astype(np.uint8))
        return v is not None and float(v @ pv) / len(v) > 0.9

    # Floating marks were written anywhere in their fill-in box, so their height above the dotted line
    # means nothing. Hang them where marks go: dash/plus/equals at mid x-height, apostrophes and quotes
    # from the top of the tall letters. One shift per character, so his own small variation is kept.
    MARK_CENTER = {"-": 0.5, "+": 0.5, "=": 0.45, "~": 0.5}
    MARK_TOP = {"'": 0.95, '"': 0.95, "*": 0.95, "^": 0.95}     # × ascender height

    def place(self, ch, rec):
        if (ch not in self.MARK_CENTER and ch not in self.MARK_TOP) or rec.get("src") == "synth":
            return rec
        if not hasattr(self, "_mark_shift"):
            self._mark_shift = {}
        if ch not in self._mark_shift:
            gs = [g for g in self.p["glyphs"].get(ch, []) if g.get("src", "real") != "synth"]
            s = 0.0
            if gs and ch in self.MARK_CENTER:
                s = self.MARK_CENTER[ch] - float(np.median([(g["top"] - g["bottom"]) / 2 for g in gs]))
            elif gs:
                s = self.MARK_TOP[ch] * self.p["means"]["asc_ratio"] - float(np.median([g["top"] for g in gs]))
            self._mark_shift[ch] = s
        s = self._mark_shift[ch]
        return dict(rec, top=rec["top"] + s, bottom=rec.get("bottom", 0.0) - s) if s else rec

    DIAG = set("AVWXYZvwxyzkK27/\\<>%&")

    def tidy(self, a, rec, ch, strength, size_strength=None):
        """Extra-neat: straighten this letter's own lean and size it to its exact class height."""
        m, fix, tgt = self.p["means"], 1.0, None
        if ch in "ij" or not ch.isalnum():
            tgt = None
        elif ch in XH or ch in DESC:
            tgt = 1.0
        elif ch in ASC:
            tgt = m["asc_ratio"]
        elif ch in CAPS:
            tgt = m["cap_ratio"]
        elif ch in DIGITS:
            tgt = m["digit_ratio"]
        if strength > 0 and ch.isalnum() and ch not in self.DIAG:
            from hw_build import deslant
            a255 = (np.clip(a, 0, 1) * 255).astype(np.uint8)
            deg = deslant(a255)[1]
            if 0 < abs(deg) <= 14:
                a = deslant(a255, fixed=deg * strength)[0].astype(np.float32) / 255.0
                cols = np.where(a.max(0) > 0.05)[0]
                if len(cols):
                    a = a[:, max(0, cols[0] - 2):cols[-1] + 3]   # drop the blank margin the shear added
        if strength > 0 and ch.isalnum() and ch not in DESC and ch != "j":       # sit exactly on the line
            rows = np.where(a.max(1) > 0.05)[0]
            if len(rows):
                exact = float(rows[-1] + 1) / rec["src_xh"]
                rec = dict(rec, top=rec["top"] + (exact - rec["top"]) * strength)
        if tgt and rec.get("top", 0) > 0.2:
            fix = clip((tgt / rec["top"]) ** (strength if size_strength is None else size_strength), 0.75, 1.33)
        return a, fix, rec

    def lookalike(self, g, ch):
        """True if this version of ch looks more like a different letter (case partner excluded)."""
        if not hasattr(self, "_look"):
            self._look = {}
            self._rare = set(jload(DATA / "rare.json", []) or [])   # his, but only occasionally
        if g["id"] in self._rare or any(p in self._rare for p in str(g["id"]).split(":")):
            return True
        if g["id"] not in self._look:
            from hw_synth import _vec
            tpl = self.templates()
            v = _vec((self.img(g) * 255).astype(np.uint8))
            res = False
            peers = [_vec((self.img(x) * 255).astype(np.uint8)) for x in self.p["glyphs"].get(ch, [])
                     if x["id"] != g["id"] and x.get("src", "real") != "synth"]
            peers = [x for x in peers if x is not None]
            if v is not None and len(peers) >= 2:
                n = len(v)
                mine = np.mean(peers, 0)                 # his OTHER versions of this letter (leave-one-out)
                own = float(v @ mine) / (n * max(1e-6, mine.std()))
                other = max((float(v @ m) / (n * max(1e-6, m.std())) for o, m in tpl.items()
                             if o.isalnum() and o.lower() != ch.lower()), default=-9)
                res = other > own
            self._look[g["id"]] = res
        return self._look[g["id"]]

    def typical(self, g, ch):
        """0..1: how close this version is to his usual shape of that character."""
        if not hasattr(self, "_typ"):
            self._typ = {}
        if g["id"] not in self._typ:
            from hw_synth import _vec
            tpl = self.templates().get(ch)
            v = _vec((self.img(g) * 255).astype(np.uint8))
            self._typ[g["id"]] = 1.0 if tpl is None or v is None else \
                clip(float(v @ tpl) / (len(v) * max(1e-6, tpl.std())), 0.0, 1.0)
        return self._typ[g["id"]]

    def note(self, gid, ch, problem):
        self.fixes.append((ch, problem))
        if gid and not gid.startswith("g"):
            self.flags[gid] = self.flags.get(gid, 0) + 1

    def save_qa(self):
        if not self.flags:
            return
        f = jload(DATA / "qa_flags.json", {}) or {}
        for k, v in self.flags.items():
            f[k] = f.get(k, 0) + v
        jsave(DATA / "qa_flags.json", f)

    def morph(self, rec1, rec2, t, rng, jitter):
        if not hasattr(self, "_morpher"):
            from hw_morph import Morpher
            self._morpher = Morpher()
        return self._morpher.make(rec1, self.img(rec1), rec2, self.img(rec2), t, rng, jitter)

    def pick(self, pool, n, task, key):
        last = task.recent.get(key, [])
        ws = []
        for g in pool:
            w = math.exp(-((g.get("neat", 0.5) - n) / 0.2) ** 2) * (0.3 + g.get("conf", 1.0)) + 1e-3  # rushed shapes go in rushed words
            if len(key) == 1:
                w *= math.exp(-12 * getattr(task, "prec", 0.0) * (1 - self.typical(g, key)))  # neat = his usual shape
                if self.lookalike(g, key):     # e.g. his f-like t: fine sometimes, rare when neat
                    w *= 0.28 * (1 - getattr(task, "prec", 0.0)) + 0.03
            if g["id"] in last[-1:] and len(pool) > 1:
                w *= 0.02
            elif g["id"] in last[-4:]:
                w *= 0.35
            ws.append(w)
        g = task.rng.choices(pool, weights=ws)[0]
        task.recent.setdefault(key, []).append(g["id"])
        return g

    def fallback(self, ch, xh_px):
        """Character not in the library yet: draw it with a handwriting font, sized to match."""
        self.fallback_used.add(ch)
        size = max(8, int(xh_px / 0.42))
        if size not in self._font:
            self._font[size] = ImageFont.truetype(str(FALLBACK_FONT), size)
        f = self._font[size]
        x0, y0, x1, y1 = f.getbbox(ch, anchor="ls")
        im = Image.new("L", (max(1, x1 - x0) + 4, max(1, y1 - y0) + 4), 0)
        ImageDraw.Draw(im).text((2 - x0, 2 - y0), ch, font=f, fill=255, anchor="ls")
        a = np.asarray(im, dtype=np.float32) / 255
        return a, dict(top=(2 - y0) / xh_px, w=im.width / xh_px, src_xh=xh_px)


# ------------------------------------------------------------------ drawing
def transform(a, scale, rot_deg, slant_deg, top_px, warp, rng, wide=1.0):
    """Scale, slant (shear about baseline), rotate and lightly warp a glyph.
    Returns (image, baseline_row, left_col_of_origin)."""
    h, w = a.shape
    pad = int(0.6 * max(h, w) * scale) + 4
    k = math.tan(math.radians(slant_deg))
    by = top_px  # baseline row in source
    # affine: scale, shear around baseline, rotate around baseline-left
    S = np.array([[scale * wide, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float64)
    Sh = np.array([[1, -k, k * by * scale], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    c, s = math.cos(math.radians(rot_deg)), math.sin(math.radians(rot_deg))
    oy = by * scale
    R = np.array([[c, s, -s * oy], [-s, c, oy - c * oy], [0, 0, 1]], dtype=np.float64)
    T = np.array([[1, 0, pad], [0, 1, pad], [0, 0, 1]], dtype=np.float64)
    M = (T @ R @ Sh @ S)[:2]
    W, H = int(w * scale * wide + 2 * pad), int(h * scale + 2 * pad)
    out = cv2.warpAffine(a, M, (W, H), flags=cv2.INTER_LINEAR)
    if warp > 0:
        amp = warp * by * scale
        sig = max(2.0, 0.35 * by * scale)
        dx = cv2.GaussianBlur(np.float32(np.random.default_rng(rng.randrange(1 << 30)).normal(0, 1, (H, W))), (0, 0), sig)
        dy = cv2.GaussianBlur(np.float32(np.random.default_rng(rng.randrange(1 << 30)).normal(0, 1, (H, W))), (0, 0), sig)
        norm = max(1e-6, np.abs(dx).max(), np.abs(dy).max())
        gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        out = cv2.remap(out, (gx + dx / norm * amp).astype(np.float32), (gy + dy / norm * amp).astype(np.float32),
                        cv2.INTER_LINEAR)
    return out, pad + oy, pad


def stamp(canvas, img, x, y):
    H, W = canvas.shape
    h, w = img.shape
    x, y = int(round(x)), int(round(y))
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 > x0 and y1 > y0:
        canvas[y0:y1, x0:x1] += img[y0 - y:y1 - y, x0 - x:x1 - x]


# ------------------------------------------------------------------ on-the-go corrections
# Look-alikes that are common look-alike pairs: never "corrected" (e.g. his t can look like f).
SIMILAR = {"t": "f+", "f": "t", "l": "1I|i!", "1": "lI|7", "I": "l1|", "o": "0Oa", "O": "0o", "0": "oO",
           "s": "5S", "S": "s5", "5": "sS", "z": "2Z", "Z": "2z", "2": "zZ", "c": "Ce(", "C": "c(", "u": "vUn",
           "v": "uVy", "n": "hu", "h": "n", "a": "dqo", "d": "a", "e": "c", "i": "jl!", "j": "i", "g": "9q",
           "9": "gq", "q": "g9a", "b": "6", "6": "b", "x": "X+", "k": "K", "w": "W", "m": "M", "p": "P", "y": "Y"}
MULTI = set("ij!?:;%=\"")


def clean_strays(a, ch):
    """Remove small specks floating away from the letter (not for characters made of separate parts)."""
    if ch in MULTI or not ch.isalnum():
        return a
    m = (a > 0.25).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 2:
        return a
    big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    bx, by, bw, bh, ba = st[big]
    pad = 0.25 * max(a.shape)
    out = a.copy()
    for j in range(1, n):
        if j == big:
            continue
        x, y, w, h, ar = st[j]
        far = x > bx + bw + pad or x + w < bx - pad or y > by + bh + pad or y + h < by - pad
        if ar < 0.08 * ba and far:
            out[(lab == j) & (a < 0.99) | (lab == j)] = 0
    return out


def word_pieces(word_text, w, lib, xh_px):
    """Return list of (alpha, meta, kind) for one word. kind: new | copy | solo | whole.
    Every letter is checked before it is used; a failing one is swapped for another version."""
    t = w.t
    prof = lib.p
    pool = prof["words"].get(word_text)
    if pool and len(word_text) >= 2 and t.rng.random() < t.p_whole * (1.4 - w.n) * (1 - max(t.tidy, t.even_size)):
        rec = lib.pick(pool, w.n, t, "W:" + word_text)
        return [(lib.img(rec), rec, "whole")]
    out = []
    for ch in word_text:
        pool = prof["glyphs"].get(ch) or []
        for tier in ("real", "derived", "synth"):
            tp = [g for g in pool if g.get("src", "real") == tier]
            if tp:
                pool = tp
                break
        if not pool:
            a, rec = lib.fallback(ch, xh_px)
            out.append((a, rec, "solo"))
            continue
        tried, piece = set(), None
        for attempt in range(5):
            cand = [g for g in pool if g["id"] not in tried and g["id"] not in lib.bad] or pool
            rec = lib.pick(cand, w.n, t, ch)
            tried.add(rec["id"])
            if len(cand) >= 2 and t.rng.random() < t.p_new:
                others = [g for g in cand if g["id"] != rec["id"]]
                same = [g for g in others if abs(g.get("neat", 0.5) - rec.get("neat", 0.5)) <= 0.25]
                others = same or others                    # blend like with like: neat+neat, rushed+rushed
                if t.prec > 0.3 and not lib.lookalike(rec, ch):
                    others = [g for g in others if not lib.lookalike(g, ch)] or others
                wts = [math.exp(-((g.get("neat", 0.5) - w.n) / 0.25) ** 2) + 0.02 for g in others]
                rec2 = t.rng.choices(others, weights=wts)[0]       # partner matches the word's neatness too
                jit = t.rng.uniform(0.2, 1.0) * (0.6 + 0.8 * (1 - w.n)) * t.chaos
                tt = t.rng.uniform(-t.reach, 1 + t.reach)
                tt = min(1 - 0.2 * t.prec, max(0.2 * t.prec, tt))
                res = lib.morph(rec, rec2, tt, t.rng, jit * t.damp)
                if res is None:
                    continue
                a, mrec, kind = res[0], res[1], "new"
            else:
                a, mrec, kind = lib.img(rec), rec, ("solo" if len(pool) < 2 else "copy")
            a = clean_strays(a, ch)
            piece = (a, mrec, kind)
            problem = lib.check(ch, a, mrec, len(pool))
            if not problem and lib.twin(ch, a):
                problem = "same as the letter next to it"
                lib.fixes.append((ch, problem))
                continue                      # not the version's fault: just draw again, don't flag it
            if not problem:
                break
            lib.note(None if kind == "new" else rec["id"], ch, problem)
        if piece is None:
            piece = (lib.img(rec), rec, "copy")
        lib.remember(ch, piece[0])
        out.append(piece)
    lib.remember(" ", None)                   # a space between words breaks "next to each other"
    return out


def layout_region(text, task, lib, W_px, H_px, xh_px, first_x, first_base, line_gap, dpi_scale):
    """Draw text into a W x H canvas. Lines drift (slope), sag or hump (curvature), get uneven
    spacing and a creeping left margin; words bob along an AR(1) baseline. Returns (canvas, fits)."""
    canvas = np.zeros((int(H_px), int(W_px)), dtype=np.float32)
    rng = task.rng
    words = text.split()

    def new_line():
        return (math.tan(math.radians(clip(task.line_slope + rng.gauss(0, task.slope_sd), -0.8, 0.8))),
                rng.gauss(0, task.curv_sd), rng.random() < 0.45)

    lpad = 0.45 * xh_px                   # room for the slant/lean of a line's first letter (it was being clipped)
    x, base = first_x + lpad, first_base
    line_start_x = x
    slope, curv, squeeze = new_line()
    fits = True
    for wi, wt in enumerate(words):
        w = task.word(wi / max(1, len(words) - 1))
        pieces = word_pieces(wt, w, lib, xh_px)
        rendered = []
        for pi, (a, rec, kind) in enumerate(pieces):
            L = w.letter()
            sc = xh_px / rec["src_xh"] * L["scale"]
            if task.even_size > 0 and kind != "whole" and pi < len(wt):
                a, fix, rec = lib.tidy(a, rec, wt[pi], task.tidy, task.even_size)
                sc *= fix
            if kind != "whole" and pi < len(wt):
                rec = lib.place(wt[pi], rec)
            wd = task.wide * clip(1 + rng.gauss(0, 0.03), 0.92, 1.08)
            wf = {"whole": 0.8, "solo": 2.2, "copy": 1.2, "new": 0.6}[kind]   # exact copies get bent more
            img, by, bx = transform(a, sc, L["rot"] * (0.5 if kind == "whole" else 1), w.slant,
                                    rec["top"] * rec["src_xh"], L["warp"] * wf, rng, wd)
            rendered.append((img, by, bx, a.shape[1] * sc * wd, L))
        for k in range(len(rendered) - 1):        # thin letters (i, l, I, 1, !) never get swallowed by neighbours
            if rendered[k][3] < 0.45 * xh_px or rendered[k + 1][3] < 0.45 * xh_px:
                rendered[k][4]["gap"] = max(rendered[k][4]["gap"], 0.12)
        ww = sum(r[3] for r in rendered) + sum(r[4]["gap"] * xh_px for r in rendered[:-1])
        if x > line_start_x and x + ww > W_px - 0.3 * xh_px:
            if getattr(task, "ruled", False):   # lined paper: every line sits on the next rule
                k = round((base - first_base) / line_gap) + 1
                base = first_base + k * line_gap + rng.gauss(0, 0.05) * xh_px * task.damp
            else:
                base += line_gap * (1 + rng.gauss(0, task.spacing_cv))
            x = lpad + abs(rng.gauss(0, task.margin_sd)) * xh_px
            line_start_x = x
            slope, curv, squeeze = new_line()
        if base + 0.9 * xh_px > H_px:
            fits = False
        u = (x - line_start_x) / max(1.0, W_px - line_start_x)
        dev = w.base * xh_px - (x - line_start_x) * slope + curv * xh_px * 4 * u * (1 - u)
        yb = base + clip(dev, -0.6 * xh_px, 0.6 * xh_px)   # never drift into the neighbouring line
        cx = x
        for img, by, bx, ink_w, L in rendered:
            stamp(canvas, img, cx - bx, yb + L["base"] * xh_px - by)
            cx += ink_w + L["gap"] * xh_px
        gap = task.word_gap(w.n)
        if squeeze and cx > 0.8 * W_px:      # running out of room: words bunch up near the edge
            gap *= 0.8 + 0.15 * w.n
        x = cx + gap * xh_px
    task._lines_used = int(round((base - first_base) / max(1e-6, line_gap))) + 1
    return canvas, fits


def pencil_dark(canvas, dpi, seed=None, tone=100):
    """Graphite-on-paper shading, after Sousa & Buchanan's observational pencil model: paper is a
    height field ("tooth"); graphite catches on the peaks first, firmer pressure fills the valleys,
    and crossing strokes add up. Light pressure -> speckled, broken edges; firm -> near solid.
    canvas: stroke coverage (glyph cores ~1, overlaps >1). Returns darkness 0..255 (float32)."""
    rng = np.random.default_rng(seed)
    h, w = canvas.shape
    px_mm = dpi / 25.4

    def octave(s_mm, ax=1.0):
        n = rng.standard_normal((h, w)).astype(np.float32)
        n = cv2.GaussianBlur(n, (0, 0), sigmaX=max(0.4, s_mm * px_mm * ax), sigmaY=max(0.4, s_mm * px_mm))
        return (n - n.mean()) / (n.std() + 1e-6)

    # paper tooth: fine fibre grain (~0.03-0.08 mm) plus faint horizontal fibres
    tooth = 1 / (1 + np.exp(-1.7 * (0.6 * octave(0.035) + 0.28 * octave(0.08) + 0.12 * octave(0.04, 5.0))))
    a = np.clip(canvas, 0, 1.4)
    shape = np.clip((a - 0.16) / 0.34, 0, 1)
    shape = shape * shape * (3 - 2 * shape)          # crisp stroke outline from the soft upscaled glyph
    hand = 1 + 0.08 * octave(1.5)                    # slow drift of hand pressure across the page
    p = shape * (0.72 + 0.28 * a) * hand             # keeps the real pressure texture from the photo
    dep = np.clip((p * 1.3 - (1 - tooth)) / 0.38, 0, 1)
    dark = dep * (0.86 + 0.14 * tooth) * (255 - tone) * (1 + 0.12 * np.clip(a - 1, 0, 0.4) / 0.4)
    return np.clip(dark, 0, 255).astype(np.float32)


def ink_to_rgba(canvas, tone=100):
    """canvas 0..1 -> RGBA graphite: one even gray (core ~ tone/255), real texture kept."""
    a = np.clip(canvas, 0, 1)
    dark = 255 - tone
    rgba = np.zeros(canvas.shape + (4,), dtype=np.uint8)
    rgba[..., :3] = (tone - 8, tone - 8, tone - 4)
    rgba[..., 3] = (a * min(255, dark / (255 - tone + 1e-6) * 255 * (dark / 255))).clip(0, 255).astype(np.uint8)
    rgba[..., 3] = (a * 255 * min(1.0, (255 - tone) / (255 - (tone - 8)))).astype(np.uint8)
    return rgba


def apply_weight(canvas, weight, xh_px):
    """weight < 0 thins strokes, > 0 thickens (fraction of an x-height)."""
    if abs(weight) < 1e-3:
        return canvas
    r = max(1, int(round(abs(weight) * xh_px * 0.08)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.erode(canvas, k) if weight < 0 else cv2.dilate(canvas, k)


def render_region(text, task, lib, w_pt, h_pt, dpi, xh_pt, first_x_pt=0, first_base_pt=None, line_gap=2.9,
                  weight=0.0):
    s = dpi / 72.0
    for shrink in (1.0, 0.93, 0.86, 0.8, 0.74):
        xh_px = xh_pt * s * shrink
        fb = (first_base_pt * s) if first_base_pt is not None else 1.9 * xh_px
        st = task.rng.getstate()
        canvas, fits = layout_region(text, task, lib, w_pt * s, h_pt * s, xh_px, first_x_pt * s, fb,
                                     (line_gap or task.line_gap_auto) * xh_px, s)
        if fits:
            break
        task.rng.setstate(st)
        task.rng.random()
    return apply_weight(canvas, weight, xh_px), fits, shrink


def load_profile():
    prof = jload(DATA / "profile.json")
    if not prof:
        sys.exit("No profile yet: import photos, label them, then run hw_build.py")
    return prof


# ------------------------------------------------------------------ commands
def cmd_text(a):
    prof = load_profile()
    rng = random.Random(a.seed)
    task = Task(prof, rng, a.neatness, load_style(a.style))
    lib = Library(prof)
    w_pt = a.width_in * 72
    canvas, fits, _ = render_region(a.text, task, lib, w_pt, a.height_in * 72, a.dpi, a.xh_pt, line_gap=a.line_gap,
                                    weight=a.weight)
    ys = np.where(canvas.max(1) > 0.05)[0]
    if ys.size:
        canvas = canvas[max(0, ys[0] - 20):ys[-1] + 20]
    dark = pencil_dark(canvas, a.dpi, a.seed, a.tone)
    pen = a.ink.lower() != "pencil"
    if pen:                                       # default black pen: solid ink, no gray
        dark = np.where(dark > 0.45 * (255 - a.tone), 255.0, 0.0)
    rgb = {"blue pen": (24, 44, 140), "blue": (24, 44, 140), "red pen": (170, 25, 30), "red": (170, 25, 30),
           "green pen": (20, 105, 50), "green": (20, 105, 50)}.get(a.ink.lower())
    if rgb and not a.transparent:
        m = (dark / 255.0)[..., None]
        im = Image.fromarray((255 * (1 - m) + np.array(rgb) * m).astype(np.uint8), "RGB")
    elif a.transparent:
        rgba = np.zeros(dark.shape + (4,), np.uint8)
        rgba[..., :3] = rgb or ((0, 0, 0) if pen else (66, 66, 70))
        rgba[..., 3] = np.clip(dark / (255 - (0 if pen else 66)) * 255, 0, 255).astype(np.uint8)
        im = Image.fromarray(rgba, "RGBA")
    else:
        im = Image.fromarray((255 - dark).astype(np.uint8), "L")
    im.save(a.out)
    lib.save_qa()
    if lib.fixes:
        print(f"auto-corrected {len(lib.fixes)} letter(s) on the go: " + ", ".join(f"{c}:{w}" for c, w in lib.fixes[:8]))
    print(f"wrote {a.out}  task neatness={task.neat:.2f}  chaos={task.chaos:.2f}  new-shape share={task.p_new:.2f}  style={task.pers.src}  locked={task.style or '-'}" + (f"  fallback chars: {''.join(sorted(lib.fallback_used))}" if lib.fallback_used else ""))


def loose_leaf(dpi, kind="college", draw=True, print_ready=False):
    """A sheet of 3-hole loose-leaf paper (RGB). Returns (image, rule geometry in points).
    draw=False: blank white page with the same geometry, for printing onto REAL loose-leaf."""
    pw, ph = 612, 792
    s = dpi / 72
    gap = 20.25 if kind != "wide" else 24.75            # college 9/32", wide 11/32"
    if not draw:
        top = 99.0
        ys = [top + k * gap for k in range(int((ph - top - 22) // gap) + 1)]
        return (Image.new("RGB", (int(pw * s), int(ph * s)), (255, 255, 255)),
                dict(ys=ys, gap=gap, margin_x=90.0, page_w=pw))
    rng = np.random.default_rng(7)
    paper = np.full((int(ph * s), int(pw * s), 3), (255, 255, 255) if print_ready else (252, 251, 246), np.float32)
    if not print_ready:                       # on screen: faint paper tint + fibre; for printing: pure white
        fib = cv2.GaussianBlur(rng.normal(0, 1, paper.shape[:2]).astype(np.float32), (0, 0), 1.2 * s / 4)
        paper += (fib / max(1e-6, np.abs(fib).max()) * 2.5)[..., None]
    im = Image.fromarray(paper.clip(0, 255).astype(np.uint8), "RGB")
    d = ImageDraw.Draw(im)
    top = 99.0
    ys = [top + k * gap for k in range(int((ph - top - 22) // gap) + 1)]
    lw = max(1, int(0.7 * s))
    for y in ys:
        d.line((0, int(y * s), int(pw * s), int(y * s)), fill=(160, 196, 228), width=lw)
    mx = 90.0
    d.line((int(mx * s), 0, int(mx * s), int(ph * s)), fill=(232, 150, 155), width=max(1, int(0.9 * s)))
    for hy in ([] if print_ready else (ph / 2 - 306, ph / 2, ph / 2 + 306)):   # 3-hole punch, 4.25" apart (print: punch real ones)
        r = 9 * s
        cx, cy = 27 * s, hy * s
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(214, 214, 210), outline=(190, 190, 186), width=max(1, int(s)))
    return im, dict(ys=ys, gap=gap, margin_x=mx, page_w=pw)


def cmd_calib(a):
    """Alignment test for printing onto real loose-leaf: thin lines where the writing's rules are, plus mm scales."""
    from reportlab.pdfgen import canvas as rl
    _, R = loose_leaf(72, a.paper, draw=False)
    c = rl.Canvas(a.out, pagesize=(612, 792))
    c.setLineWidth(0.35)
    for y in R["ys"]:
        c.line(110, 792 - y, 600, 792 - y)
    c.line(R["margin_x"], 792 - 40, R["margin_x"], 792 - 760)
    mm = 72 / 25.4
    c.setFont("Helvetica", 6)
    for k in range(-10, 11):                       # vertical mm scale next to rule 3, horizontal next to the margin
        y = 792 - R["ys"][3] + k * mm
        c.line(48, y, 58 if k % 5 else 64, y)
        if k % 5 == 0:
            c.drawString(66, y - 2, f"{-k:+d}mm")
        x = R["margin_x"] + k * mm
        c.line(x, 792 - 30, x, 792 - (38 if k % 5 else 44))
    c.setFont("Helvetica", 8)
    c.drawString(110, 792 - 22, "Print at 100% (Actual size) on a sheet of loose-leaf. Lines should sit ON the blue lines;")
    c.drawString(110, 792 - 31, "the short vertical line at top should sit on the red margin. If not, read the mm scales and tell Claude.")
    c.showPage()
    c.save()
    print(f"wrote {a.out}")


def pen_mark(kind, x, y, w, h, dpi, rng, stroke_px, pts_in=None):
    """A hand-drawn mark as an ink canvas (same units as letters): "ellipse" circles the box (x, y, w, h)
    the way a hand does (wobbly, tilted, overshooting its start); "check" is a check mark filling the box;
    "line" follows pts_in (points, page coords). Returns (canvas, left_pt, top_pt)."""
    s, k = dpi / 72, 4                                       # k: supersampling for smooth edges
    if kind == "ellipse":
        rx, ry = w / 2 * 1.22 + 2.0, h / 2 * 1.32 + 1.6       # enclose the box's corners, with breathing room
        cx, cy = x + w / 2, y + h / 2
        a0, sweep = rng.uniform(0, 2 * math.pi), 2 * math.pi * rng.uniform(1.06, 1.16)
        tilt, ph = rng.gauss(0, 0.05), [rng.uniform(0, 6.3) for _ in range(2)]
        pts = []
        for i in range(181):
            t = i / 180
            a = a0 + sweep * t
            r = (1 + 0.06 * math.sin(2 * a + ph[0]) + 0.035 * math.sin(3 * a + ph[1])
                 + 0.015 * math.sin(5 * a + ph[0] * 2)) * (1 + 0.09 * t)       # lopsided, end spirals out
            ex, ey = rx * r * math.cos(a), ry * r * math.sin(a)
            pts.append((cx + ex * math.cos(tilt) - ey * math.sin(tilt), cy + ex * math.sin(tilt) + ey * math.cos(tilt)))
    elif kind == "check":
        p0 = (x + w * rng.uniform(0.0, 0.08), y + h * rng.uniform(0.5, 0.6))
        p1 = (x + w * rng.uniform(0.3, 0.4), y + h * rng.uniform(0.95, 1.0))
        p2 = (x + w * rng.uniform(0.92, 1.0), y + h * rng.uniform(0.0, 0.08))
        bow = rng.uniform(0.04, 0.1) * w
        pts = [(p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t) for t in np.linspace(0, 1, 12)]
        for t in np.linspace(0, 1, 30)[1:]:                  # long up-stroke, bowed a little
            mx, my = p1[0] + (p2[0] - p1[0]) * t, p1[1] + (p2[1] - p1[1]) * t
            pts.append((mx + bow * math.sin(math.pi * t) * 0.5, my + bow * math.sin(math.pi * t)))
    else:
        pts = [tuple(p) for p in pts_in]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    pad = 3 * stroke_px / s + 2
    left, top = min(xs) - pad, min(ys) - pad
    W, H = int((max(xs) - left + pad) * s) + 1, int((max(ys) - top + pad) * s) + 1
    im = Image.new("L", (W * k, H * k), 0)
    dr = ImageDraw.Draw(im)
    P = [((px - left) * s * k, (py - top) * s * k) for px, py in pts]
    for i in range(len(P) - 1):
        t = i / max(1, len(P) - 2)
        wd = stroke_px * k * (0.8 + 0.3 * math.sin(math.pi * t)) * rng.uniform(0.95, 1.05)   # lighter at the ends
        dr.line([P[i], P[i + 1]], fill=255, width=max(1, int(round(wd))))
        r = wd / 2
        dr.ellipse([P[i + 1][0] - r, P[i + 1][1] - r, P[i + 1][0] + r, P[i + 1][1] + r], fill=255)
    a = np.asarray(im.resize((W, H), Image.LANCZOS), dtype=np.float32) / 255.0
    return a, left, top


def raster_pages(pdf, dpi):
    d = Path(tempfile.mkdtemp())
    subprocess.run(["pdftoppm", "-r", str(dpi), "-gray", "-png", str(pdf), str(d / "pg")], check=True)
    return [Image.open(p).convert("L") for p in sorted(d.glob("pg-*.png"), key=lambda p: int(re.findall(r"\d+", p.stem)[-1]))]


def cmd_spec(a):
    spec = jload(a.spec)
    prof = load_profile()
    dpi = spec.get("dpi", 600)
    rng = random.Random(spec.get("seed"))
    task = Task(prof, rng, spec.get("neatness"), load_style(spec.get("style")))
    lib = Library(prof)
    paper = spec.get("paper")
    if paper:
        pages, rules = [], None
        for _ in range(spec.get("pages", 1)):
            pg, rules = loose_leaf(dpi, paper, not spec.get("writing_only"), spec.get("print_ready", False))
            pages.append(pg)
        task.ruled = True
        if spec.get("margin_pt") is not None:      # left edge of the writing, in points
            rules["margin_x"] = spec["margin_pt"] - 6
        elif spec.get("writing_only"):              # plain paper: normal 0.75" margin, no hole clearance
            rules["margin_x"] = 48.0
        if "xh_pt" not in spec:            # he fills more of the line when writing extra neatly
            spec["xh_pt"] = round((0.31 + 0.05 * task.prec + 0.01 * task.tidy) * rules["gap"], 2)
        for r in spec.get("regions", []):          # snap each region to the rules
            if r.get("ruled", True):
                r.setdefault("x", rules["margin_x"] + 6)
                r.setdefault("w", rules["page_w"] - rules["margin_x"] - 34)
                first = rules["ys"][r.get("start_line", 0)]
                r["y"] = first - 2.2 * r.get("xh_pt", spec.get("xh_pt", 6.2))
                r["first_baseline"] = first - r["y"] + 0.35   # letters sit ON the rule
                r["line_gap"] = rules["gap"] / r.get("xh_pt", spec.get("xh_pt", 6.2))
                r.setdefault("h", rules["ys"][-1] - r["y"] + 8)
    elif spec.get("base_pdf"):
        pages = raster_pages(spec["base_pdf"], dpi)
    else:
        pw, ph = spec.get("page_size_pt", [612, 792])
        pages = [Image.new("L", (int(pw * dpi / 72), int(ph * dpi / 72)), 255) for _ in range(spec.get("pages", 1))]
    tone = spec.get("tone", 100)
    warn = []
    s = dpi / 72

    # INK (default black pen). "pencil" = graphite shader; "pen"/"black pen" = solid black; "blue pen",
    # "red pen", "green pen" or an [r, g, b] list = solid coloured ink. "pure_black": false also means pencil.
    ink_name = spec.get("ink", "pencil" if spec.get("pure_black") is False else "pen")
    INKS = {"pen": None, "black pen": None, "black": None, "blue pen": (24, 44, 140), "blue": (24, 44, 140),
            "red pen": (170, 25, 30), "red": (170, 25, 30), "green pen": (20, 105, 50), "green": (20, 105, 50)}
    pencil_ink = str(ink_name).lower() == "pencil"
    ink_rgb = tuple(ink_name) if isinstance(ink_name, list) else INKS.get(str(ink_name).lower())

    def put(pidx, canvas, x_pt, y_pt, seed_k):
        """Shade a block of handwriting in the chosen ink and lay it onto page pidx at (x_pt, y_pt)."""
        pg = pages[pidx]
        ink = pencil_dark(canvas, dpi, None if spec.get("seed") is None else spec["seed"] + seed_k, tone)
        if not pencil_ink:                         # pen: solid ink, no gray -> no printer dot pattern
            ink = np.where(ink > 0.45 * (255 - tone), 255.0, 0.0).astype(np.float32)
        x0, y0 = int(x_pt * s), int(y_pt * s)
        cx0, cy0 = max(0, -x0), max(0, -y0)       # a mark poking past the page edge is cut, not shifted
        ink, x0, y0 = ink[cy0:, cx0:], x0 + cx0, y0 + cy0
        if ink_rgb and pg.mode != "RGB":
            pg = pg.convert("RGB")
        pa = np.asarray(pg, dtype=np.float32).copy()
        h, w = ink.shape
        h, w = min(h, pa.shape[0] - y0), min(w, pa.shape[1] - x0)
        if h <= 0 or w <= 0:
            return
        dk = ink[:h, :w] if pa.ndim == 2 else ink[:h, :w, None]
        if ink_rgb:                                # coloured pen: tint toward the ink colour where it lands
            m = dk / 255.0
            pa[y0:y0 + h, x0:x0 + w] = np.minimum(pa[y0:y0 + h, x0:x0 + w],
                                                  pa[y0:y0 + h, x0:x0 + w] * (1 - m) + np.array(ink_rgb, np.float32) * m)
        else:
            # pencil/black ink darkens what is under it, never lightens print
            pa[y0:y0 + h, x0:x0 + w] = np.minimum(pa[y0:y0 + h, x0:x0 + w], pa[y0:y0 + h, x0:x0 + w] * (1 - dk / 255.0))
            pa[y0:y0 + h, x0:x0 + w] = np.minimum(pa[y0:y0 + h, x0:x0 + w], 255 - dk)
        pages[pidx] = Image.fromarray(pa.clip(0, 255).astype(np.uint8), pg.mode)

    for i, r in enumerate(spec.get("regions", [])):
        canvas, fits, shrink = render_region(
            r["text"], task, lib, r["w"], r["h"], dpi, r.get("xh_pt", spec.get("xh_pt", 7.0)),
            r.get("first_line_x", 0), r.get("first_baseline"), r.get("line_gap", spec.get("line_gap")),
            r.get("weight", spec.get("weight", 0.0)))
        if not fits:
            warn.append(f"region {i + 1} overflows (text too long for its box)")
        elif shrink < 1:
            warn.append(f"region {i + 1} shrunk to {shrink:.0%} to fit")
        put(r.get("page", 1) - 1, canvas, r["x"], r["y"], i)

    # MARKS: hand-drawn circles around things, check marks in boxes, free lines/arrows (same ink as the writing)
    for j, m in enumerate(spec.get("marks", [])):
        mr = random.Random((spec.get("seed") or 0) * 1000 + j)
        stroke_px = m.get("stroke_pt", 0.1 * spec.get("xh_pt", 6.0)) * s
        canvas, lx, ty = pen_mark(m.get("type", "ellipse"), m.get("x", 0), m.get("y", 0), m.get("w", 0), m.get("h", 0),
                                  dpi, mr, stroke_px, m.get("points"))
        put(m.get("page", 1) - 1, canvas, lx, ty, 900 + j)

    # FLOW: a document of blocks (headings, dash items with hanging indent, blank lines) written down the
    # rules, continuing onto new sheets. A block is never split across pages.
    if spec.get("flow") and paper:
        xh = spec["xh_pt"]
        gap, ys = rules["gap"], rules["ys"]
        left, right = rules["margin_x"] + 6, rules["page_w"] - (44 if spec.get("writing_only") else 26)
        page, line = 0, spec.get("start_line", 0)
        for k, blk in enumerate(spec["flow"], start=100):
            if blk.get("blank"):
                line += int(blk["blank"])
                continue
            x_text = left + blk.get("indent", 0) + blk.get("hang", 0)
            for attempt in (0, 1):
                if line >= len(ys):
                    page, line = page + 1, 0
                while page >= len(pages):
                    pages.append(loose_leaf(dpi, paper, not spec.get("writing_only"), spec.get("print_ready", False))[0])
                y = ys[line] - 2.2 * xh
                st = task.rng.getstate()
                canvas, fits, _ = render_region(blk["text"], task, lib, right - x_text, ys[-1] - y + 10, dpi, xh, 0,
                                                ys[line] - y + 0.35, gap / xh, 0.0)
                used = getattr(task, "_lines_used", 1)
                need = 0                     # a heading stays with the block under it (no orphan at a page foot)
                nxt = spec["flow"][k - 99] if k - 99 < len(spec["flow"]) else None
                if (line + used <= len(ys) and not blk.get("marker") and nxt and nxt.get("text")
                        and not nxt.get("blank")):
                    st2 = task.rng.getstate()
                    xn = left + nxt.get("indent", 0) + nxt.get("hang", 0)
                    render_region(nxt["text"], task, lib, right - xn, ys[-1] - y + 10, dpi, xh, 0,
                                  ys[line] - y + 0.35, gap / xh, 0.0)
                    need = min(getattr(task, "_lines_used", 1), len(ys) // 2)
                    task.rng.setstate(st2)
                if line + used + need <= len(ys) or attempt == 1:
                    break
                task.rng.setstate(st)
                page, line = page + 1, 0                 # doesn't fit: start the block on a fresh sheet
            if blk.get("marker"):
                mc, _, _ = render_region(blk["marker"], task, lib, blk.get("hang", 14) + 4, 4 * xh, dpi, xh, 0,
                                         ys[line] - y + 0.35, gap / xh, 0.0)
                put(page, mc, left + blk.get("indent", 0), y, k + 500)
            put(page, canvas, x_text, y, k)
            line += used
    out = Path(spec.get("out_pdf", "handwritten.pdf"))
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas as rl
    if spec.get("offset_pt"):                # nudge to line up with a particular brand of loose-leaf
        dx, dy = spec["offset_pt"]
        for k, pg in enumerate(pages):
            sh = Image.new(pg.mode, pg.size, (255, 255, 255) if pg.mode == "RGB" else 255)
            sh.paste(pg, (int(round(dx * dpi / 72)), int(round(dy * dpi / 72))))
            pages[k] = sh
    if spec.get("writing_only"):
        pages = [pg if ink_rgb else pg.convert("L") for pg in pages]   # grayscale unless the ink is coloured
    pw_pt, ph_pt = pages[0].width * 72 / dpi, pages[0].height * 72 / dpi
    c = rl.Canvas(str(out), pagesize=(pw_pt, ph_pt))
    for pg in pages:
        c.drawImage(ImageReader(pg), 0, 0, width=pw_pt, height=ph_pt)
        c.showPage()
    c.save()
    if spec.get("out_png_prefix"):
        for k, pg in enumerate(pages, 1):
            pg.save(f"{spec['out_png_prefix']}-{k}.png")
    lib.save_qa()
    if lib.fixes:
        print(f"auto-corrected {len(lib.fixes)} letter(s) on the go")
    print(f"wrote {out}  pages={len(pages)}  task neatness={task.neat:.2f}")
    if lib.fallback_used:
        print(f"fallback font used for: {''.join(sorted(lib.fallback_used))}  (add these via the gap sheet)")
    for w_ in warn:
        print("warning:", w_)


def cmd_lines(a):
    """Print every text line of a PDF with its box in points (origin top-left)."""
    out = subprocess.run(["pdftotext", "-bbox-layout", a.pdf, "-"], check=True, capture_output=True, text=True).stdout
    page = 0
    for m in re.finditer(r'<page |<line xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">(.*?)</line>', out, re.S):
        if m.group(0).startswith("<page"):
            page += 1
            continue
        words = re.findall(r">([^<]+)</word>", m.group(5))
        print(f"p{page} x={float(m.group(1)):.1f} y={float(m.group(2)):.1f} x2={float(m.group(3)):.1f} "
              f"y2={float(m.group(4)):.1f}  {' '.join(words)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("text")
    t.add_argument("text")
    t.add_argument("--out", required=True)
    t.add_argument("--width-in", type=float, default=6.5)
    t.add_argument("--height-in", type=float, default=9.0)
    t.add_argument("--xh-pt", type=float, default=7.0, help="x-height in points (~7 = normal school handwriting)")
    t.add_argument("--line-gap", type=float, default=None, help="line spacing in x-heights (default: from his pages)")
    t.add_argument("--neatness", type=float, default=None, help="0 sloppy .. 1 neat; default: random like a real task")
    t.add_argument("--seed", type=int, default=None)
    t.add_argument("--dpi", type=int, default=600)
    t.add_argument("--tone", type=int, default=100, help="pencil core gray 0-255 (100 = prints well)")
    t.add_argument("--weight", type=float, default=0.0, help="<0 thinner strokes, >0 thicker")
    t.add_argument("--transparent", action="store_true")
    t.add_argument("--ink", default="pen", help='"pen" (black, default), "pencil", "blue pen", "red pen", "green pen"')
    t.add_argument("--style", default=None, help='locked choices, e.g. "neatness=0.85 slant=8 letter_spacing=tight consistency=consistent page=IMG_3187"')
    s = sub.add_parser("spec")
    s.add_argument("spec")
    cb = sub.add_parser("calib")
    cb.add_argument("--out", required=True)
    cb.add_argument("--paper", default="college")
    l_ = sub.add_parser("lines")
    l_.add_argument("pdf")
    a = ap.parse_args()
    {"text": cmd_text, "spec": cmd_spec, "lines": cmd_lines, "calib": cmd_calib}[a.cmd](a)


if __name__ == "__main__":
    main()
