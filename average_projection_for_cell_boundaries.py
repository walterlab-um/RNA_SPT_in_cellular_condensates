import os
from os.path import join, isdir
import numpy as np
from tifffile import imread, imwrite
from random import choices
from rich.progress import track
import shutil

# Make an average projection of the condensate channale and save to a subfolder called "cell_body"

# Change the below directory to a mother folder containing subfolders for each condition. Within each subfolder, a "condensate" folder must be found containing all condensate videos in tif format.
dir_from = "/Volumes/AnalysisGG/PROCESSED_DATA/RNA_in_HOPS-Jun2023_wrapup/LiveCell-different-RNAs-in-HOPS"
# dir_from = "/Volumes/AnalysisGG/PROCESSED_DATA/RNA_in_HOPS-Jun2023_wrapup/LiveCell-different-RNAs-in-HOPS/Isotonic-1x"

# In case the dir_from folder has other unwanted subfolders
postfix = "100ms"
lst_subfolders = [f for f in os.listdir(dir_from) if f.endswith("100ms")]
# Otherwise:
# lst_subfolders = [f for f in os.listdir(dir_from)]

os.chdir(dir_from)
for subfolder in track(lst_subfolders):
    # create the "cell_body" subfolder if not exsisted yet
    if not isdir(join(subfolder, "cell_body")):
        os.mkdir(join(subfolder, "cell_body"))

    all_files_in_subfolder = [
        f for f in os.listdir(join(subfolder, "condensate")) if f.endswith(".tif")
    ]
    for fname in all_files_in_subfolder:
        video = imread(join(dir_from, subfolder, "condensate", fname))
        average_projection = np.mean(video, axis=0).astype("uint16")
        imwrite(
            join(subfolder, "cell_body", fname[:-4] + "-AveProj.tif"),
            average_projection,
            imagej=True,
        )
