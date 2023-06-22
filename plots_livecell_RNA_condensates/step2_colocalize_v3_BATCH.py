import GGspt
import os, sys
from os.path import join, isdir
import pandas as pd
from rich.progress import track
from tkinter import filedialog as fd

##############################
# Settings
path = "/Volumes/AnalysisGG/PROCESSED DATA/2022May-LiveCellCleanup"
r_search = 10  # cut off for RNA search, unit:pixel
lst_folders = [
    "THOR_2x",
    "THOR_1x",
    "FL_2x",
    "miR21_2x",
    "FL_wCHX_2x",
    "THORdel_2x",
    "THORdel_1x",
    "miRcxcr4_2x",
]

##############################
# Main

for folder in lst_folders:
    lst_fnames_RNA = [
        f for f in os.listdir(join(path, folder)) if f.endswith("-right.csv")
    ]
    common_names = [f.split("-right.csv")[0] for f in lst_fnames_RNA]
    lst_fnames_con = [f + "-left.csv" for f in common_names]
    os.chdir(join(path, folder))
    for i in track(range(len(common_names)), description=folder):
        df_RNA = pd.read_csv(lst_fnames_RNA[i], dtype=float)
        df_con = pd.read_csv(lst_fnames_con[i], dtype=float)
        path_save = common_names[i] + "_colocalization.csv"
        GGspt.GG_colocalize(df_RNA, df_con, r_search, path_save)
