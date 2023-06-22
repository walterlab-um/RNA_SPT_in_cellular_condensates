import GGspt
import os, sys
from os.path import dirname, basename, join
import pandas as pd
import numpy as np
from tkinter import filedialog as fd
from rich.progress import track


##############################
# Settings
# print("Select a folder containing (1)colocalization results in csv format")
# folderpath = fd.askdirectory(
# initialdir="/Volumes/AnalysisGG/PROCESSED DATA/2022May-LiveCellCleanup"
# )
# print("The data folder is: ", folderpath)
path = "/Volumes/AnalysisGG/PROCESSED_DATA/2022May-LiveCellCleanup"
lst_folders = [
    "THOR_1x",
    "THORdel_1x",
]

# This parameter should match TrackMate/KNIME parameter, it's the maximum number of missing frames in a trajectory that can be tolerate, or say two trajectories seperated with less than this number can be joined as one trajectory
max_gap = 2


##############################
# Main
os.chdir(path)
for folder in lst_folders:
    lst_fnames_coloc = [
        f for f in os.listdir(folder) if f.endswith("_colocalization.csv")
    ]
    col1_fname = []
    col2_dwelltimes = []
    os.chdir(folder)
    for fname in track(lst_fnames_coloc, description=folder):
        df_coloc = pd.read_csv(fname, dtype=float)

        # Interaction defined as within (1) estimated condensate radius; (2) 1.5 times estimated R; (3) a fixed radius of 5 pixels; (4) a fixed radius of 7.5 pixels
        condition = 4
        if condition == 1:
            df_coloc = df_coloc[df_coloc.distance < df_coloc.est_R].sort_values(
                by=["RNA_trackID", "t"]
            )
        elif condition == 2:
            df_coloc = df_coloc[df_coloc.distance < df_coloc.est_R * 1.5].sort_values(
                by=["RNA_trackID", "t"]
            )
        elif condition == 3:
            df_coloc = df_coloc[df_coloc.distance < 5].sort_values(
                by=["RNA_trackID", "t"]
            )
        elif condition == 4:
            df_coloc = df_coloc[df_coloc.distance < 7.5].sort_values(
                by=["RNA_trackID", "t"]
            )

        dwelltimes = GGspt.calc_dwelltime(df_coloc.t.to_numpy())

        dwelltimes_filtered = dwelltimes[dwelltimes >= 1]

        col2_dwelltimes.append(dwelltimes_filtered)
        col1_fname.append(np.repeat(fname.split("_coloc")[0], dwelltimes_filtered.size))

    df_save = pd.DataFrame(
        {"filename": np.hstack(col1_fname), "dwelltimes": np.hstack(col2_dwelltimes)}
    )
    os.chdir(path)
    fname_save = folder + "-condition" + str(condition) + "-dwelltimes.csv"
    df_save.to_csv(fname_save, index=False)
