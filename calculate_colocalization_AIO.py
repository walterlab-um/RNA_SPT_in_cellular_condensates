import os
from os.path import join, dirname
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
from rich.progress import track

pd.options.mode.chained_assignment = None  # default='warn'

# AIO: All in one format


# scalling factors for physical units
um_per_pixel = 0.117
s_per_frame = 0.1
print(
    "Scaling factors: s_per_frame = "
    + str(s_per_frame)
    + ", um_per_pixel = "
    + str(um_per_pixel)
)

# Change the below directory to a mother folder containing subfolders for each condition. Within each subfolder, a "condensate" folder must be found containing all condensate videos in tif format.
dir_from = "/Volumes/AnalysisGG/PROCESSED_DATA/RNA_in_HOPS-Jun2023_wrapup/LiveCell-different-RNAs-in-HOPS"
# dir_from = "/Volumes/AnalysisGG/PROCESSED_DATA/RNA_in_HOPS-Jun2023_wrapup/LiveCell-different-RNAs-in-HOPS/Isotonic-1x"

# In case the dir_from folder has other unwanted subfolders
postfix = "100ms"
lst_subfolders = [f for f in os.listdir(dir_from) if f.endswith("100ms")]
# Otherwise:
# lst_subfolders = [f for f in os.listdir(dir_from)]


# Output file columns
columns = [
    "fname_RNA",
    "fname_condensate",
    "fname_cell",
    "RNA_trackID",
    "t",
    "x",
    "y",
    "InCondensate",
    "condensateID",
    "distance_to_center_nm",
    "distance_to_edge_nm",
]


def list_like_string_to_polygon(list_like_string):
    # example list_like_string structure of polygon coordinates: '[[196, 672], [196, 673], [197, 673], [198, 673], [199, 673], [199, 672], [198, 672], [197, 672]]'
    list_of_xy_string = list_like_string[2:-2].split("], [")

    coords_roi = []
    for xy_string in list_of_xy_string:
        x, y = xy_string.split(", ")
        coords_roi.append((int(x), int(y)))

    polygon_output = Polygon(coords_roi)

    return polygon_output


def list_like_string_to_xyt(list_like_string):
    # example list_like_string structure of xyt: '[0, 1, 2, 3]'
    list_of_xyt_string = list_like_string[1:-1].split(", ")
    lst_xyt = []
    for xyt_string in list_of_xyt_string:
        lst_xyt.append(float(xyt_string))

    return lst_xyt


lst_rows_of_df = []
os.chdir(dir_from)
## loop through each subfolder/experiment condition
for subfolder in lst_subfolders:
    all_fname_cell_body = [
        f for f in os.listdir(join(subfolder, "cell_body")) if f.endswith(".txt")
    ]
    all_fname_RNA_AIO = [
        f for f in os.listdir(join(subfolder, "RNA")) if f.startswith("SPT_results_AIO")
    ]
    all_fname_condensate_AIO = [
        f
        for f in os.listdir(join(subfolder, "condensate"))
        if f.startswith("condensates_AIO")
    ]

    ## loop through each FOV
    for fname_RNA_AIO in all_fname_RNA_AIO:
        df_RNA = pd.read_csv(join(subfolder, "RNA", fname_RNA_AIO))

        # find the fnames for cells within the current FOV
        fnames_cells_in_current_FOV = []
        for f in all_fname_cell_body:
            if fname_RNA_AIO[16:].split("-RNA")[0] in f:
                fnames_cells_in_current_FOV.append(f)
                break
        if len(fnames_cells_in_current_FOV) == 0:
            print("There's no cells in current FOV.")
            break

        # load the condensates AIO file for the current FOV
        for f in all_fname_condensate_AIO:
            if fname_RNA_AIO[16:].split("-RNA")[0] in f:
                fname_condensate_AIO = f
                break
        if "fname_condensate_AIO" not in globals():
            print("There's no condensate AIO file for the current FOV.")
            break
        df_condensate = pd.read_csv(join(subfolder, "condensate", fname_condensate_AIO))

        # process RNA tracks one by one
        for trackID in df_RNA["trackID"]:
            current_track = df_RNA[df_RNA["trackID"] == trackID]
            lst_x = list_like_string_to_xyt(current_track["list_of_x"].squeeze())
            lst_y = list_like_string_to_xyt(current_track["list_of_y"].squeeze())
            lst_t = list_like_string_to_xyt(current_track["list_of_t"].squeeze())

            # process each position in track one by one
            for i in len(lst_t):
                t = lst_t[i]
                x = lst_x[i]
                y = lst_y[i]

                # Save
                new_row = [
                    fname_RNA_AIO,
                    fname_condensate_AIO,
                    "fname_cell",
                    trackID,
                    t,
                    x,
                    y,
                    "InCondensate",
                    "condensateID",
                    "distance_to_center_nm",
                    "distance_to_edge_nm",
                ]
                lst_rows_of_df.append(new_row)

df_save = pd.DataFrame.from_records(
    lst_rows_of_df,
    columns=columns,
)
fname_save = join(dirname(fpath), "SPT_results_AIO-pleaserename.csv")
df_save.to_csv(fname_save, index=False)
