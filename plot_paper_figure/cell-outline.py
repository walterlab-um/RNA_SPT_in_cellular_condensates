# cell_outline_to_png.py
# -------------------------------------------------------------
import numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

ROOT          = "/Users/esumrall/Desktop/RNA-in-HOPS_condensates/4h/FL-2x"
os.chdir(ROOT)
# ---------------- USER-PARAMS ----------------
TXT_FILE      = "20220508-FLmRNA_2x_FOV-6-cell-1.txt"
OUT_PNG       = "cell_outline.png"
LINE_COLOR    = "black"     # any Matplotlib colour
LINE_WIDTH    = 4         # pixels
CANVAS_SIZE   = (10, 8)      # inches
PAD_PX        = 1          # white frame around outline
# -------------------------------------------------------------

# 1. load X,Y pairs
xy = np.loadtxt(TXT_FILE, dtype=float)          # [file:4]
if not np.allclose(xy[0], xy[-1]):
    xy = np.vstack([xy, xy[0]])                 # close polygon

# 2. figure + transparent canvas
fig, ax = plt.subplots(figsize=CANVAS_SIZE, dpi=300)
fig.patch.set_alpha(0)                          # transparent background
ax.axis("off"); ax.set_aspect("equal")

# 3. draw outline
ax.plot(xy[:, 0], xy[:, 1],
        color=LINE_COLOR, linewidth=LINE_WIDTH)

# 4. tight limits + padding
xmin, xmax = xy[:, 0].min(), xy[:, 0].max()
ymin, ymax = xy[:, 1].min(), xy[:, 1].max()
ax.set_xlim(xmin - PAD_PX, xmax + PAD_PX)
ax.set_ylim(ymax + PAD_PX, ymin - PAD_PX)

# 5. export
plt.savefig(OUT_PNG, transparent=True, dpi=300, bbox_inches="tight")
plt.close()
print(f"✅  Saved transparent outline   →  {OUT_PNG}")
