#!/usr/bin/env python3
"""The fill-in kit: printable sheets a person writes on, then photographs or scans and sends back.

  hw_kit.py make [--quick] --out kit.pdf      make the kit to print and fill in
  hw_kit.py read FILE [FILE ...] [--kit CODE] read filled pages (photos and/or scanned PDFs, any order)
  hw_kit.py status                            what's been read so far, library size, what's missing

One person per install: whoever this skill belongs to. Everything stays in data/ on this machine.

Kit pages
  * Character pages: every letter, digit and mark in its own row of boxes. Left boxes = careful
    writing, right boxes = fast writing (so the model learns both ends of how they write).
  * Copy pages (3): a printed paragraph to copy line for line onto ruled lines, once NEAT, once
    NORMAL, once FAST. The text is known, so every word is labeled automatically, and the pages also
    teach how their writing moves on a page (spacing, drift, size wander, overlap).

Every page has 4 black corner squares (to straighten the photo) and a row of small code squares at
the top that say which page it is, so photos can come back in any order, rotated or not.
"""
import argparse
import random
import string
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas as rl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hwcore import DATA, flatten_light, jload, jsave, load_photo, rel_w  # noqa: E402

PW, PH = letter
M, FID = 36, 22                     # page margin, corner square size (pt)
BOX_W, BOX_H = 52, 40               # character box (pt)
BASE_FR = 0.64                      # dotted baseline at this fraction of the box height
ROW_GAP, LABEL_W = 6, 34
S = 2400 / PW                       # reading scale: px per pt (same page width the importer uses)
CHARS = string.ascii_lowercase + string.ascii_uppercase + string.digits + ".,?!'-:;()$%/&+=\"@#*"
NBITS = 9                           # [1, 0] sync + 6 page bits + parity
BIT, BIT_GAP = 8, 6
RULE0, RULE_GAP, N_RULES = 318, 26, 10
WX0, WX1 = 70, PW - M - 4           # writing area on copy pages

COPY = [
    ("NEAT", 0.88, "your NEATEST writing, like a final copy", [
        "The quick brown fox jumps over the lazy",
        "dog. Pack my box with five dozen liquor",
        "jugs! On 10/14 at 3:45, Mike and Zoe",
        "bought 7 jars of jam for $5.99 (20% off).",
        "Was that a great deal? Yes; 2 + 2 = 4,",
        "and 6 - 1 = 5. 'Quiet,' said Vera & Kim.",
        "Grab the black wizard's quilt, Jeff,",
        "and write \"Exam 8\" on the 9th."]),
    ("NORMAL", None, "your normal everyday writing, like homework", [
        "Sphinx of black quartz, judge my vow.",
        "How vexingly quick daft zebras jump!",
        "Bright vixens jump; dozy fowl quack.",
        "We met at Room 214 on Friday, June 6,",
        "then left at 7:30 for the Oak Street",
        "game. Liam, Uma, Xavier and Yara won",
        "18 to 9, so everyone got pizza (yum)!",
        "Next up: Tuesday's quiz on chapter 3."]),
    ("FAST", 0.25, "FAST, like rushed notes in class (don't slow down)", [
        "notes - ch 5: the war ended in 1865 b/c",
        "the North had more people, factories,",
        "and railroads. Lincoln + Grant pushed;",
        "Lee gave up at Appomattox (April 9).",
        "Key terms: Reconstruction, 13th Amend,",
        "Freedmen's Bureau, Black Codes, KKK.",
        "Quiz Wed?? study pages 112-130 & do",
        "the review sheet. Due 4/18 = Thursday!"]),
]


# ------------------------------------------------------------------ geometry shared by make + read
def fid_centers():
    o = M + FID / 2
    return [(o, o), (PW - o, o), (PW - o, PH - o), (o, PH - o)]      # tl, tr, br, bl (top-left origin)


def bit_boxes():
    x0 = PW / 2 - (NBITS * BIT + (NBITS - 1) * BIT_GAP) / 2
    y = M + FID / 2 - BIT / 2
    return [(x0 + k * (BIT + BIT_GAP), y) for k in range(NBITS)]


def page_bits(idx):
    b = [(idx >> k) & 1 for k in range(6)]
    return [1, 0] + b + [sum(b) % 2]


# ------------------------------------------------------------------ make
def ensure_data():
    for sub_ in ("sources", "labels", "library", "kits", "report"):
        (DATA / sub_).mkdir(parents=True, exist_ok=True)
    for f, v in (("rejects.json", []), ("rare.json", []), ("qa_flags.json", {}), ("style_defaults.json", {})):
        if not (DATA / f).exists():
            jsave(DATA / f, v)


def status():
    prof = jload(DATA / "profile.json") or {}
    kits = sorted(p.stem for p in (DATA / "kits").glob("*.json")) if (DATA / "kits").exists() else []
    srcs = sorted(p.name for p in (DATA / "sources").iterdir()) if (DATA / "sources").exists() else []
    print(f"kits made:  {', '.join(kits) or 'none yet (hw_kit.py make --out kit.pdf)'}")
    print(f"pages read: {len(srcs)} source(s) {' '.join(srcs)}")
    if prof:
        print(f"model:      built, {sum(len(v) for v in prof.get('glyphs', {}).values())} letter versions, "
              f"missing: {''.join(prof.get('missing', [])) or 'nothing'}")
    else:
        print("model:      not built yet (read the filled kit first)")


def make(out, quick=False):
    ensure_data()
    code = "".join(random.choice("ABCDEFGHJKMNPQRSTUVWXYZ23456789") for _ in range(4))
    per_row = 6 if quick else 8                     # half careful, half fast
    gap_x = (PW - 2 * M - LABEL_W - per_row * BOX_W) / (per_row + 1)
    top0 = M + FID + 44
    rows_pp = int((PH - top0 - M - FID - 6) // (BOX_H + ROW_GAP))
    chars = list(CHARS)
    char_pages = [chars[i:i + rows_pp] for i in range(0, len(chars), rows_pp)]
    n_pages = len(char_pages) + len(COPY)
    c = rl.Canvas(str(out), pagesize=letter)
    pages = []

    def frame(idx, title, sub):
        for (fx, fy) in fid_centers():
            c.rect(fx - FID / 2, PH - fy - FID / 2, FID, FID, stroke=0, fill=1)
        for (bx, by), bit in zip(bit_boxes(), page_bits(idx)):
            if bit:
                c.rect(bx, PH - by - BIT, BIT, BIT, stroke=0, fill=1)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(M + FID + 10, PH - M - FID - 14, f"{title}   (kit {code}, page {idx + 1} of {n_pages})")
        c.setFont("Helvetica", 9)
        c.drawString(PW - M - FID - 170, PH - M - FID - 14, "Name:")
        c.setStrokeGray(0.5)
        c.setLineWidth(0.5)
        c.line(PW - M - FID - 142, PH - M - FID - 15, PW - M - FID - 4, PH - M - FID - 15)
        c.setFont("Helvetica", 8.5)
        c.drawString(M + FID + 10, PH - M - FID - 27, sub)

    for pi, pchars in enumerate(char_pages):
        frame(pi, "Handwriting kit: characters",
              "Dark pen or pencil. Sit each character on the dotted line, at your normal size. "
              f"Left {per_row // 2} boxes: careful. Right {per_row // 2}: fast.")
        boxes = []
        for ri, ch in enumerate(pchars):
            yt = top0 + ri * (BOX_H + ROW_GAP)
            c.setFillGray(0.45)
            c.setFont("Helvetica", 13)
            c.drawCentredString(M + LABEL_W / 2, PH - yt - BOX_H * BASE_FR, ch)
            for k in range(per_row):
                x = M + LABEL_W + gap_x + k * (BOX_W + gap_x)
                c.setStrokeGray(0.6 if k < per_row // 2 else 0.4)
                c.setLineWidth(0.6)
                c.rect(x, PH - yt - BOX_H, BOX_W, BOX_H, stroke=1, fill=0)
                c.setStrokeGray(0.82)
                c.setDash(1, 2.5)
                c.line(x + 3, PH - yt - BOX_H * BASE_FR, x + BOX_W - 3, PH - yt - BOX_H * BASE_FR)
                c.setDash()
                boxes.append(dict(ch=ch, neat=k < per_row // 2, x=x, y=yt))
        c.setFillGray(0)
        c.showPage()
        pages.append(dict(kind="chars", boxes=boxes))

    for ci, (tag, neat, how, lines) in enumerate(COPY):
        idx = len(char_pages) + ci
        frame(idx, f"Handwriting kit: copy ({tag})",
              f"Copy the text onto the lines below, LINE FOR LINE (line 1 on line 1...), in {how}.")
        c.setFont("Helvetica", 8.5)
        c.drawString(M + FID + 10, PH - M - FID - 39, "Use the whole width like normal, skip nothing, and don't worry about mistakes.")
        y = M + FID + 70
        c.setStrokeGray(0.7)
        c.rect(WX0 - 6, PH - y - 8 * 15 - 6, WX1 - WX0 + 12, 8 * 15 + 18, stroke=1, fill=0)
        c.setFont("Helvetica", 11)
        for k, t in enumerate(lines):
            c.setFillGray(0.5)
            c.drawString(WX0 - 2 - 14, PH - y - 10 - k * 15, f"{k + 1}")
            c.setFillGray(0.15)
            c.drawString(WX0 + 4, PH - y - 10 - k * 15, t)
        c.setStrokeColorRGB(0.55, 0.72, 0.95)        # light blue rules (dropped as colour when reading)
        c.setLineWidth(0.7)
        for k in range(N_RULES):
            ry = RULE0 + k * RULE_GAP
            c.line(WX0 - 4, PH - ry, WX1, PH - ry)
            if k < len(lines):
                c.setFillGray(0.6)
                c.setFont("Helvetica", 8)
                c.drawString(M + FID - 4, PH - ry + 2, f"{k + 1}")
        c.setFillGray(0)
        c.showPage()
        pages.append(dict(kind="copy", tag=tag, neat=neat, lines=lines))
    c.save()
    jsave(DATA / "kits" / f"{code}.json", dict(code=code, quick=quick, pages=pages))
    print(f"kit {code}: {n_pages} pages ({len(char_pages)} character pages + {len(COPY)} copy pages) -> {out}")
    return code


# ------------------------------------------------------------------ read
def find_fiducials(gray):
    """The 4 black corner squares, in full-resolution pixel coordinates (found on a small copy for speed).
    RETR_LIST, not EXTERNAL: in a photo the squares sit inside the paper, which sits inside the table."""
    k = min(1.0, 1600 / max(gray.shape))
    small = cv2.resize(gray, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else gray
    small = cv2.GaussianBlur(small, (3, 3), 0)
    paper = float(np.percentile(small, 75))                  # most of the frame is paper
    m = (small < 0.45 * paper).astype(np.uint8)              # squares are far darker than the paper around them
    cnts, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    H, W = small.shape
    cands = []
    for cn in cnts:
        a = cv2.contourArea(cn)
        x, y, w, h = cv2.boundingRect(cn)
        if a < 1e-4 * H * W or a > 0.01 * H * W or not (0.7 < w / max(h, 1) < 1.4) or a < 0.8 * w * h:
            continue
        cands.append((x + w / 2, y + h / 2))
    if len(cands) < 4:
        return None
    corners = [(0, 0), (W, 0), (W, H), (0, H)]
    pts = [min(cands, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2) for cx, cy in corners]
    return np.float32(pts) / k if len(set(pts)) == 4 else None


def read_bits(warped):
    d = 255 - flatten_light(warped[:int((M + FID + 12) * S)])     # just the top strip: fast
    out = []
    for bx, by in bit_boxes():
        x0, y0 = int((bx + 1.5) * S), int((by + 1.5) * S)
        x1, y1 = int((bx + BIT - 1.5) * S), int((by + BIT - 1.5) * S)
        out.append(1 if d[y0:y1, x0:x1].mean() > 110 else 0)
    if out[:2] != [1, 0] or sum(out[2:8]) % 2 != out[8]:
        return None
    return sum(b << k for k, b in enumerate(out[2:8]))


def straighten(gray):
    """Find the corner squares, warp to page coordinates, and work out which kit page it is.
    Tries all four rotations (photos come back sideways or upside down)."""
    size = (int(PW * S), int(PH * S))
    dst = np.float32([[x * S, y * S] for x, y in fid_centers()])
    g = gray
    for _ in range(4):
        pts = find_fiducials(g)
        if pts is not None:
            w = cv2.warpPerspective(g, cv2.getPerspectiveTransform(pts, dst), size, flags=cv2.INTER_CUBIC,
                                    borderValue=255)
            idx = read_bits(w)
            if idx is not None:
                return w, idx
        g = cv2.rotate(g, cv2.ROTATE_90_CLOCKWISE)
    return None, None


def load_inputs(files):
    """Photos (jpg/heic/png) and scanned PDFs -> list of (name, gray image)."""
    out = []
    for f in files:
        f = Path(f)
        if f.suffix.lower() == ".pdf":
            d = Path(tempfile.mkdtemp())
            subprocess.run(["pdftoppm", "-r", "300", "-gray", "-png", str(f), str(d / "p")], check=True)
            for p in sorted(d.glob("p-*.png")):
                out.append((f"{f.name} p{p.stem.split('-')[-1].lstrip('0') or '0'}", np.asarray(Image.open(p).convert("L"))))
        else:
            out.append((f.name, cv2.cvtColor(load_photo(str(f)), cv2.COLOR_BGR2GRAY)))
    return out


def clean_box(dark, x0, y0, x1, y1):
    """Ink of one filled box with the box border, dotted guide and specks removed (None = empty)."""
    crop = dark[y0:y1, x0:x1]
    t = max(60.0, cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])
    mask = (crop > t).astype(np.uint8)
    nl, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(nl, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= 14
    mask = keep[lab].astype(np.uint8)
    Hc, Wc = mask.shape
    hl = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (int(0.5 * Wc), 1)))
    vl = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(0.6 * Hc))))
    band = np.zeros_like(mask)
    bw = max(6, int(0.03 * 72 * S))
    band[:, :bw] = band[:, -bw:] = 1
    band[:bw, :] = band[-bw:, :] = 1
    edge = (hl | vl) & cv2.dilate(band, np.ones((5, 5), np.uint8))
    nl, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for j in range(1, nl):
        x, y, w, h, _ = st[j]
        if (x <= 1 or y <= 1 or x + w >= Wc - 1 or y + h >= Hc - 1) and band[lab == j].mean() > 0.6:
            edge[lab == j] = 1
    mask &= ~(cv2.dilate(edge, np.ones((3, 3), np.uint8)) > 0).astype(np.uint8)
    if mask.sum() < 30:
        return None
    zone = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
    return np.where(zone & (crop > 22), crop, 0).astype(np.uint8)


def read_chars(kit, found):
    """All character pages -> one 'gap' source (g###) of labeled single characters."""
    ids = [int(p.name[1:4]) for p in (DATA / "sources").glob("[dg][0-9][0-9][0-9]") if p.is_dir()]
    doc = f"g{(max(ids) if ids else 0) + 1:03d}"
    d = DATA / "sources" / doc
    (d / "words").mkdir(parents=True, exist_ok=True)
    recs, labels, n = [], {}, 0
    for idx, (name, w) in sorted(found.items()):
        dark = 255 - flatten_light(w)
        for b in kit["pages"][idx]["boxes"]:
            inset = 5 * S
            x0, y0 = int(b["x"] * S + inset), int(b["y"] * S + inset)
            x1, y1 = int((b["x"] + BOX_W) * S - inset), int((b["y"] + BOX_H) * S - inset)
            ink = clean_box(dark, x0, y0, x1, y1)
            if ink is None:
                continue
            n += 1
            wid = f"w{n:03d}"
            Image.fromarray(ink, "L").save(d / "words" / f"{wid}.png")
            recs.append(dict(id=wid, line=0, bbox=[x0, y0, x1, y1], baseline=float(BOX_H * BASE_FR * S - inset),
                             neat=0.85 if b["neat"] else 0.3))
            labels[wid] = b["ch"]
    jsave(d / "meta.json", dict(doc=doc, kind="gap", sheet=kit["code"], est_h=0.35 * BOX_H * S, words=recs,
                                photos=[found[i][0] for i in sorted(found)]))
    jsave(DATA / "labels" / f"{doc}.json", dict(doc=doc, name=f"kit {kit['code']} characters", neatness=None, words=labels))
    return doc, n


def _align(meas, toks):
    """Match measured words to the known text, both in reading order. meas = [(rule, width)], toks =
    [(line, token)]. One DP over the whole page (so a merged or split word can't shift the rest), with a
    width cost after fitting one scale per page, and a penalty for crossing lines. Returns {i: j} for
    confident matches only: right line AND size within ~30%."""
    e = np.array([sum(rel_w(ch) for ch in t) for _, t in toks], float)
    w = np.array([x for _, x in meas], float)
    if not len(w) or not len(e):
        return {}
    sc = float(np.median(w)) / float(np.median(e))
    n, m = len(w), len(e)
    D = np.full((n + 1, m + 1), 1e9)
    D[0, :] = np.arange(m + 1) * 0.8
    D[:, 0] = np.arange(n + 1) * 0.8
    bk = {}
    cost = lambda i, j: abs(np.log(w[i] / (sc * e[j]))) + (0.0 if meas[i][0] == toks[j][0] else 0.9)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j], bk[i, j] = min((D[i - 1, j - 1] + cost(i - 1, j - 1), "m"), (D[i - 1, j] + 0.8, "u"),
                                    (D[i, j - 1] + 0.8, "l"))
    i, j, out = n, m, {}
    while i > 0 and j > 0:
        op = bk[i, j]
        if op == "m":
            if meas[i - 1][0] == toks[j - 1][0] and abs(np.log(w[i - 1] / (sc * e[j - 1]))) < 0.3:
                out[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif op == "u":
            i -= 1
        else:
            j -= 1
    return out


def read_copy(kit, idx, name, w, keep_color=False):
    """One copy page -> a 'page' source (d###) through the normal importer, then auto-labeled."""
    import hw_import
    page = kit["pages"][idx]
    y0, y1 = int((RULE0 - 34) * S), int((RULE0 + (N_RULES - 1) * RULE_GAP + 16) * S)
    crop = w[y0:y1].copy()
    crop[:, :int((WX0 - 8) * S)] = 255                # line numbers / margin
    crop[:, int((WX1 + 4) * S):] = 255
    tmp = Path(tempfile.mkdtemp()) / f"kit_{kit['code']}_{page['tag'].lower()}.png"
    Image.fromarray(crop, "L").save(tmp)

    from types import SimpleNamespace
    doc = hw_import.import_photo(str(tmp), SimpleNamespace(printed="none", keep_color=keep_color, no_crop=True))
    meta = jload(DATA / "sources" / doc / "meta.json")
    lab = jload(DATA / "labels" / f"{doc}.json")
    k = 2400 / crop.shape[1]                           # importer rescales to its page width (here 1.0)
    rows = {}
    for r in meta["words"]:
        bx0, by0, bx1, by1 = r["bbox"]
        yb = by1 / k / S + (RULE0 - 34)                # bottom of the word box, page pt
        rule = int(round((yb - RULE0) / RULE_GAP + 0.15))
        rows.setdefault(rule, []).append(r)
    ordered = [(rule, r) for rule in sorted(rows) if 0 <= rule < len(page["lines"])
               for r in sorted(rows[rule], key=lambda r: r["bbox"][0])]
    toks = [(li, t) for li, ln in enumerate(page["lines"]) for t in ln.split()]
    pairs = _align([(rule, r["bbox"][2] - r["bbox"][0]) for rule, r in ordered], toks)
    words = {r["id"]: "#skip" for r in meta["words"]}
    for i, j in pairs.items():
        words[ordered[i][1]["id"]] = toks[j][1]
    n_lab, n_lines = len(pairs), len({ordered[i][0] for i in pairs})
    lab["words"] = words
    lab["neatness"] = page["neat"]
    lab["name"] = f"kit {kit['code']} copy {page['tag']}"
    lab["note"] = "Auto-labeled from the known copy text; unmatched words are #skip."
    jsave(DATA / "labels" / f"{doc}.json", lab)
    meta["name"] = lab["name"]
    jsave(DATA / "sources" / doc / "meta.json", meta)
    total = sum(len(t.split()) for t in page["lines"])
    print(f"   copy {page['tag']:6s} -> {doc}: {n_lab}/{total} words labeled on {n_lines} lines")
    return doc, n_lab


def read(files, code=None, keep_color=False, build=True):
    ensure_data()
    kits = sorted((DATA / "kits").glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not kits:
        sys.exit("no kit made yet: run  hw_kit.py make --out kit.pdf")
    kit = jload(DATA / "kits" / f"{code}.json") if code else jload(kits[-1])
    if not kit:
        sys.exit(f"unknown kit {code}")
    found = {}
    for name, g in load_inputs(files):
        w, idx = straighten(g)
        if w is None or idx >= len(kit["pages"]):
            print(f"{name}: couldn't find the 4 corner squares / page code. Retake it flat, all 4 corners in view.")
            continue
        if idx in found:
            print(f"{name}: page {idx + 1} again, keeping the first copy")
            continue
        found[idx] = (name, w)
        print(f"{name} -> page {idx + 1} ({kit['pages'][idx]['kind']})")
    missing = [i + 1 for i in range(len(kit["pages"])) if i not in found]
    chars = {i: v for i, v in found.items() if kit["pages"][i]["kind"] == "chars"}
    if chars:
        doc, n = read_chars(kit, chars)
        print(f"   characters -> {doc}: {n} filled boxes")
    for i, (name, w) in sorted(found.items()):
        if kit["pages"][i]["kind"] == "copy":
            read_copy(kit, i, name, w, keep_color)
    if missing:
        print(f"not received yet: page(s) {', '.join(map(str, missing))} (send them any time; read just those later)")
    if build and found:
        here = Path(__file__).resolve().parent
        print("building the handwriting model ...")
        subprocess.run([sys.executable, str(here / "hw_layout.py")], check=True)
        subprocess.run([sys.executable, str(here / "hw_build.py")], check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--out", required=True)
    m.add_argument("--quick", action="store_true", help="6 boxes per character instead of 8")
    r = sub.add_parser("read")
    r.add_argument("files", nargs="+")
    r.add_argument("--kit", default=None, help="kit code (default: the newest kit)")
    r.add_argument("--keep-color", action="store_true", help="written in blue/coloured pen")
    r.add_argument("--no-build", action="store_true")
    sub.add_parser("status")
    a = ap.parse_args()
    if a.cmd == "status":
        status()
    elif a.cmd == "make":
        make(a.out, a.quick)
    else:
        read(a.files, a.kit, a.keep_color, not a.no_build)


if __name__ == "__main__":
    main()
