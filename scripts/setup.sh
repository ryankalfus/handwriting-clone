#!/bin/zsh
# One-time setup: private venv for the handwriting skill.
# Reuses Pillow + reportlab from the system python3; adds numpy + OpenCV (headless).
set -e
S="${0:A:h:h}"
if [[ ! -x "$S/.venv/bin/python" ]]; then
  python3 -m venv --system-site-packages "$S/.venv"
fi
"$S/.venv/bin/python" -m pip install --quiet --upgrade pip
"$S/.venv/bin/python" -m pip install --quiet numpy opencv-python-headless
"$S/.venv/bin/python" -c "import numpy, cv2, PIL, reportlab; print('ok numpy', numpy.__version__, 'cv2', cv2.__version__, 'PIL', PIL.__version__, 'reportlab', reportlab.Version)"
