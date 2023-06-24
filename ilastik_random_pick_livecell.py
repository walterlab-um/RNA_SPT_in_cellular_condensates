import os
from os.path import join
import numpy as np
from tifffile import imread, imwrite
from random import choices
from rich.progress import track
import shutil

# This script randomly picks k FOV from the whole dataset to pool a ilastik training dataset
k = 7

# Select what will be in the training dataset
# Selector: 1. direct copy tif file; 2. average projection; 3. first frame

# for HOPS condensates, do direct copy, because we need to predict condensate for each frame
# selector = 1
# for cell body, do average protection, assuming cell doesn't move for the 20 s imaging window
selector = 2
# for some cases, the first frame (before any photo-bleaching) may be needed
# selector = 3

# Change the below directory to a mother folder containing subfolders for each condition. Within each subfolder, a "condensate" folder must be found containing all condensate videos in tif format.
dir_from = "/Volumes/AnalysisGG/PROCESSED_DATA/RNA_in_HOPS-Jun2023_wrapup/LiveCell-different-RNAs-in-HOPS"

# The directory to store the ilastik training dataset.
# dir_to = "/Volumes/AnalysisGG/PROCESSED_DATA/Training-ilastik/livecell_100ms-Dcp1a_HOPS" # for HOPS condensates
dir_to = "/Volumes/AnalysisGG/PROCESSED_DATA/Training-ilastik/livecell_100ms-cell_body_byDcp1a"  # for cell body

# In case the dir_from folder has other unwanted subfolders
postfix = "100ms"
lst_subfolders = [f for f in os.listdir(dir_from) if f.endswith("100ms")]
# Otherwise:
# lst_subfolders = [f for f in os.listdir(dir_from)]

os.chdir(dir_from)
for subfolder in track(lst_subfolders):
    all_files_in_subfolder = [
        f for f in os.listdir(join(subfolder, "condensate")) if f.endswith(".tif")
    ]
    chosen_ones = choices(all_files_in_subfolder, k=k)
    for fname in chosen_ones:
        if selector == 1:
            shutil.copy(
                join(dir_from, subfolder, "condensate", fname),
                join(dir_to, fname),
            )
        elif selector == 2:
            video = imread(join(dir_from, subfolder, "condensate", fname))
            average_projection = np.mean(video, axis=0).astype("uint16")
            imwrite(join(dir_to, fname), average_projection, imagej=True)
        elif selector == 3:
            video = imread(join(dir_from, subfolder, "condensate", fname))
            first_frame = video[0, :, :].astype("uint16")
            imwrite(join(dir_to, fname), first_frame, imagej=True)
        else:
            print("Please input the right selector.")
