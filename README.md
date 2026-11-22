# handwriting-clone

A Claude Code skill that clones **your** handwriting from a printable fill-in kit, then writes anything in it. Black pen is the default; pencil and blue, red or green pen are on request.

## Install

```bash
git clone https://github.com/<you>/handwriting-clone ~/.claude/skills/handwriting-clone
~/.claude/skills/handwriting-clone/scripts/setup.sh     # private venv: numpy + OpenCV (needs python3 with Pillow + reportlab)
```

Also needs poppler (`pdftoppm`, `pdftotext`), e.g. `brew install poppler`.

## Use

Just ask Claude:
1. "Make me the handwriting kit." You get a 10-page PDF. Print it at Actual Size.
2. Fill it in with dark pen or pencil:
   - every character in its boxes: careful on the left, fast on the right
   - three short paragraphs copied line for line: neat, normal, fast
3. Send photos or a scanner-app PDF of the pages, in any order. Claude reads them, builds your model, and shows you a sheet of your letters to check.
4. "Write this in my handwriting…": notes on loose-leaf, a filled-in worksheet PDF, check marks and circles, neat or sloppy, in any ink.

## Privacy

- One person per install.
- Your samples and model live in `data/` on your machine only. They're git-ignored and never uploaded.
- The skill only writes in the handwriting learned from its own kit. It can't be pointed at anyone else's.

Scripts: `scripts/hw_kit.py` (kit make/read/status) and `scripts/hw_render.py` (writing). The rest is the engine: import, build, morph, layout.
