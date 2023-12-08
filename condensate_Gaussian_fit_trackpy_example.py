from tifffile import imread
import trackpy as tp
import os
import matplotlib.pyplot as plt

folder = "/Volumes/lsa-nwalter/Guoming_Gao_turbo/Walterlab_server/PROCESSED_DATA/RNA-in-HOPS_condensates/LiveCell-different-RNAs-in-HOPS/FL-2x-100ms/condensate"
os.chdir(folder)
fname = "20220508FLmRNA_2x_FOV_1-cropped-left.tif"

video = imread(fname)
img = video[0][100:250, 200:350]
# img = video[0]
df = tp.locate(
    img,
    5,
    # minmass=100,
    percentile=95,
)

plt.figure(figsize=(5, 5))
plt.imshow(img, cmap="gray", vmin=400, vmax=700)
plt.plot(
    [5, 5 + 1000 / 117],
    [5, 5],
    "-",
    color="w",
    lw=5,
)
plt.gca().invert_yaxis()
plt.axis("off")
plt.axis("equal")
df_plot = df[df["raw_mass"] < 1.5e4]
for x, y, r in zip(df_plot.x, df_plot.y, df_plot["size"] * 2):
    c = plt.Circle((x, y), r, color="firebrick", fill=False)
    plt.gca().add_patch(c)
plt.savefig(
    "../../trackpy_rawmass_10000_filter_HOPS_example.png",
    dpi=300,
    format="png",
    bbox_inches="tight",
)
plt.close()
