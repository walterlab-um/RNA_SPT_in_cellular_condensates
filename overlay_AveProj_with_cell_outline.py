import os
from os.path import join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread
from rich.progress import track

# Check all the subfoler "cell_body" under each condition. Plot the cell outline overlay with the average projection of condensate channel.

# Change the below directory to a mother folder containing subfolders for each condition. Within each subfolder, a "condensate" folder must be found containing all condensate videos in tif format.
dir_from = "/Volumes/AnalysisGG/PROCESSED_DATA/RNA_in_HOPS-Jun2023_wrapup/LiveCell-different-RNAs-in-HOPS"
# dir_from = "/Volumes/AnalysisGG/PROCESSED_DATA/RNA_in_HOPS-Jun2023_wrapup/LiveCell-different-RNAs-in-HOPS/Isotonic-1x"

# In case the dir_from folder has other unwanted subfolders
postfix = "100ms"
lst_subfolders = [f for f in os.listdir(dir_from) if f.endswith("100ms")]
# Otherwise:
# lst_subfolders = [f for f in os.listdir(dir_from)]

plow = 0.05  # imshow intensity percentile
phigh = 99

os.chdir(dir_from)
for subfolder in track(lst_subfolders):
    all_cell_body_outline_fname = [
        f for f in os.listdir(join(subfolder, "cell_body")) if f.endswith(".txt")
    ]
    all_FOV_fname = [
        f for f in os.listdir(join(subfolder, "cell_body")) if f.endswith(".tif")
    ]

    if len(all_cell_body_outline_fname) == 0:
        print("There's no mannual cell body outline txt files for:", subfolder)
        continue

    for FOVfname in all_FOV_fname:
        FOVprefix = FOVfname[:-25]
        cells_in_current_FOV = [
            f for f in all_cell_body_outline_fname if FOVprefix in f
        ]
        AveProj = imread(join(subfolder, "cell_body", FOVfname))

        plt.figure(dpi=600)
        # Contrast stretching
        vmin, vmax = np.percentile(AveProj, (plow, phigh))
        plt.imshow(AveProj, cmap="gray", vmin=vmin, vmax=vmax)
        # plot all cell outlines
        for cell_fname in cells_in_current_FOV:
            cell_outline_coordinates = pd.read_csv(
                join(subfolder, "cell_body", cell_fname), sep="	", header=None
            )
            x = cell_outline_coordinates.iloc[:, 0].to_numpy(dtype=int)
            y = cell_outline_coordinates.iloc[:, 1].to_numpy(dtype=int)
            plt.plot(x, y, "--", color="snow", linewidth=2)
            # still the last closing line will be missing, get it below
            xlast = [x[-1], x[0]]
            ylast = [y[-1], y[0]]
            plt.plot(xlast, ylast, "--", color="snow", linewidth=2)
            plt.text(
                x.mean(),
                y.mean(),
                cell_fname.split("cell-")[-1][:-4],
                c="red",
                weight="bold",
            )

        plt.xlim(0, AveProj.shape[0])
        plt.ylim(0, AveProj.shape[1])
        plt.tight_layout()
        plt.axis("scaled")
        plt.axis("off")
        fpath_save = join(subfolder, "cell_body", FOVprefix + ".png")
        plt.savefig(fpath_save, format="png", bbox_inches="tight", dpi=600)
        plt.close()
