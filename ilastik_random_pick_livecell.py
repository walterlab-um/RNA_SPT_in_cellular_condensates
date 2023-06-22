import os
from os.path import join, isfile
import shutil
from random import choices

# This script randomly picks 5 FOV from each condition to pool a ilastik traking dataset, to tranin a machine learning model for condensate boundary detection.

folder_from = "/Volumes/AnalysisGG/PROCESSED_DATA/RNA_SPT_in_FUS-May2023_wrapup/Condensates_AveProj"
folder_to = "/Volumes/AnalysisGG/PROCESSED_DATA/Training-ilastik/FUS_invitro_20ms"

lst_subfolders = [f for f in os.listdir(folder_from) if not isfile(f)]

os.chdir(folder_from)
for subfolder in lst_subfolders:
    all_files_in_subfolder = [f for f in os.listdir(subfolder) if f.endswith(".tif")]
    chosen_ones = choices(all_files_in_subfolder, k=5)
    for fname in chosen_ones:
        shutil.copy(join(folder_from, subfolder, fname), join(folder_to, fname))
