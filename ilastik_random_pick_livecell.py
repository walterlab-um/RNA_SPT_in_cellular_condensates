import os
from os.path import join
import numpy as np
from tifffile import imread, imwrite
from random import choices
from rich.progress import track

# This script randomly picks 7 FOV from each condition to pool a ilastik traking dataset, to tranin a machine learning model for condensate boundary detection.

dir_from = "/Volumes/AnalysisGG/PROCESSED_DATA/RNA_in_HOPS-Jun2023_wrapup/LiveCell-different-RNAs-in-HOPS"
dir_HOPS_condensate = (
    "/Volumes/AnalysisGG/PROCESSED_DATA/Training-ilastik/livecell_100ms-Dcp1a_HOPS"
)
dir_cell_body = "/Volumes/AnalysisGG/PROCESSED_DATA/Training-ilastik/livecell_100ms-cell_body_byDcp1a"

lst_subfolders = [f for f in os.listdir(dir_from) if f.endswith("100ms")]

os.chdir(dir_from)
for subfolder in track(lst_subfolders):
    all_files_in_subfolder = [
        f for f in os.listdir(subfolder) if f.endswith("left.tif")
    ]
    chosen_ones = choices(all_files_in_subfolder, k=7)
    for fname in chosen_ones:
        video = imread(join(dir_from, subfolder, fname))
        first_frame = video[0, :, :].astype("uint16")
        average_projection = np.mean(video, axis=0).astype("uint16")
        imwrite(join(dir_HOPS_condensate, fname), first_frame, imagej=True)
        imwrite(join(dir_cell_body, fname), average_projection, imagej=True)
