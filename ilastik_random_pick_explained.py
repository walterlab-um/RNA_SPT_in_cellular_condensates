import os
from os.path import join
import numpy as np
from tifffile import imread, imwrite
from random import choices
from rich.progress import track
import shutil

# This script randomly picks x FOV from the whole dataset to pool a ilastik training dataset
x = 5

# Change this directory to the folder containing images or videos to process (the whole dataset)
dir_from = "/Volumes/AnalysisGG/PROCESSED_DATA/RNA_in_HOPS-Jun2023_wrapup/LiveCell-different-RNAs-in-HOPS"
# This is a folder to store the randomly picked videos
dir_video = (
    "/Volumes/AnalysisGG/PROCESSED_DATA/Training-ilastik/livecell_100ms-Dcp1a_HOPS"
)
# This is a folder to store the average projection images of the randomly picked videos
dir_average_projection = "/Volumes/AnalysisGG/PROCESSED_DATA/Training-ilastik/livecell_100ms-cell_body_byDcp1a"


os.chdir(dir_from)

all_files_in_subfolder = [f for f in os.listdir(".") if f.endswith(".tif")]
chosen_ones = choices(all_files_in_subfolder, k=x)
for fname in chosen_ones:
    video = imread(join(dir_from, fname))
    # first_frame = video[0, :, :].astype("uint16")
    average_projection = np.mean(video, axis=0).astype("uint16")
    # imwrite(join(dir_video, fname), first_frame, imagej=True)
    shutil.copy(
        join(dir_from, fname),
        join(dir_video, fname),
    )
    imwrite(join(dir_average_projection, fname), average_projection, imagej=True)
