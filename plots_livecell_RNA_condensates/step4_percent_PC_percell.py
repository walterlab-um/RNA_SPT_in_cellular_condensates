import os, sys
from os.path import join, dirname, basename
import pandas as pd
import numpy as np
from rich.progress import track
from tkinter import filedialog as fd
from shapely.geometry import Point, Polygon

##############################
# Settings
# print(
# "Select a folder containing\n (1)colocalization results ending in '-cropped_colocalization.csv';\n (2):roi files ending in '.txt';\n (3) RNA tracks ending in '-cropped-right.csv';\n (4) condensate tracks ending in '-cropped-left.csv'"
# )
# folderpath = fd.askdirectory(
# initialdir="/Volumes/AnalysisGG/PROCESSED DATA/2022May-LiveCellCleanup"
# )
# print("The data folder is: ", folderpath)
dir_main = "/Volumes/AnalysisGG/PROCESSED DATA/2022May-LiveCellCleanup"


# A relative ratio to correct estimated radius. The interaction cut off will be ratio * est_R of each condensate. The radius estimation algorithm of TrackMate is not efficient in estimating the size of small blobs like HOPS condensates. It usually underestimate the size, and thus needs the correction ratio below.
# It turns out this only exagerate the noise! Abandon!
# ratio = 1.5
um_per_pxl = 0.117

##############################
# Main
def Main(folderpath, ratio, um_per_pxl):
    lst_roi_processed = []
    lst_percent = []
    lst_PC = []
    lst_N_total = []
    lst_N_coloc = []
    lst_A_cell = []
    lst_A_con = []

    # loop through every cell for colocalization
    lst_roi = [f for f in os.listdir(folderpath) if f.endswith(".txt")]
    for roi in track(
        lst_roi, description="Calculating coloc percent & Partition Coefficient..."
    ):
        # import data
        common_name = roi.split("-cell")[0]

        fname_RNA = common_name + "-cropped-right.csv"
        fname_con = common_name + "-cropped-left.csv"
        fname_coloc = common_name + "-cropped_colocalization.csv"

        df_RNA = pd.read_csv(join(folderpath, fname_RNA), dtype=float)
        df_con = pd.read_csv(join(folderpath, fname_con), dtype=float)
        df_coloc = pd.read_csv(join(folderpath, fname_coloc), dtype=float)

        # extract x, y, trackID for colocalization
        x_coloc = df_coloc[
            df_coloc.distance < df_coloc["est_R"] * ratio
        ].RNA_X.to_numpy()
        y_coloc = df_coloc[
            df_coloc.distance < df_coloc["est_R"] * ratio
        ].RNA_Y.to_numpy()
        RNAid_coloc = df_coloc[
            df_coloc.distance < df_coloc["est_R"] * ratio
        ].RNA_trackID.to_numpy()

        # import mask for each cell
        df_roi = pd.read_csv(join(folderpath, roi), sep="	", header=None)
        coords_roi = [tuple(row) for index, row in df_roi.iterrows()]
        mask = Polygon(coords_roi)

        # count RNAs
        RNAid_all_inmask = [
            df_RNA.trackID[idx]
            for idx in range(len(df_RNA.x))
            if Point(df_RNA.x[idx], df_RNA.y[idx]).within(mask)
        ]
        RNAid_coloc_inmask = [
            RNAid_coloc[idx]
            for idx in range(len(x_coloc))
            if Point(x_coloc[idx], y_coloc[idx]).within(mask)
        ]

        # apply a frame > 3 threshold on RNAid_coloc_inmask to ensure it's a real interaction with condensates
        unique, counts = np.unique(RNAid_coloc_inmask, return_counts=True)
        df_dwell = pd.DataFrame({"RNAid_coloc_inmask": unique, "dwell": counts})
        N_coloc = df_dwell[df_dwell["dwell"] >= 3]["RNAid_coloc_inmask"].size
        N_total = np.unique(RNAid_all_inmask).size

        # Partition Coefficient formula is
        # (N_coloc / (N_total - N_coloc)) * ((A_cell - A_con) / A_con)
        A_cell = (um_per_pxl ** 2) * mask.area
        lst_current_A_con = []
        for conID in np.unique(df_con.trackID):
            current_con = df_con[df_con["trackID"] == conID]
            if Point(current_con.x.mean(), current_con.y.mean()).within(mask):
                current_r = (current_con.estDiameter.mean() / 2) * um_per_pxl * ratio
                current_A_con = np.pi * (current_r ** 2)
                lst_current_A_con.append(current_A_con)
        A_con = np.sum(lst_current_A_con)
        PC = (N_coloc / (N_total - N_coloc)) * ((A_cell - A_con) / A_con)

        # save results
        lst_roi_processed.append(roi)
        lst_percent.append(100 * N_coloc / N_total)
        lst_PC.append(PC)
        lst_N_total.append(N_total)
        lst_N_coloc.append(N_coloc)
        lst_A_cell.append(A_cell)
        lst_A_con.append(A_con)

    df_out = pd.DataFrame(
        {
            "roi": lst_roi_processed,
            "Percentage of Colocalization": lst_percent,
            "Partition Coefficient": lst_PC,
            "N_total in roi": lst_N_total,
            "N_coloc in roi": lst_N_coloc,
            "A_cell, um^2": lst_A_cell,
            "A_con, um^2": lst_A_con,
        },
        dtype=object,
    )

    fname_save = join(
        dirname(folderpath),
        basename(folderpath)
        + "-ratio"
        + str(ratio)
        + "-All_coloc_percent_PC_percellroi.csv",
    )

    df_out.to_csv(fname_save, index=False)


lst_folders = [
    "miRcxcr4_2x",
]
for folder in lst_folders:
    print(folder)
    Main(join(dir_main, folder), 1, 0.117)
