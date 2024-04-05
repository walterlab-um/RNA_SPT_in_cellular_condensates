import os
from os.path import join, dirname
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points
from shapely import distance
from rich.progress import track
from tkinter import filedialog as fd
import tkinter as tk

pd.options.mode.chained_assignment = None  # default='warn'

# AIO: All in one format
# This script gives the collocalization information for every RNA position over time.
# This version is for live cell data, including cell boundaries


# scalling factors for physical units
print("Type in the pixel size in nm:")
nm_per_pixel = float(input())
print("Scaling factors: nm_per_pixel = " + str(nm_per_pixel))

print(
    "Please choose a folder containing 3 subfolders: RNA, condensate, cell_body_mannual:"
)
root = tk.Tk()
root.withdraw()  # Hide the main window
folder_path = fd.askdirectory()
os.chdir(folder_path)

# parameters
interaction_cutoff = 10  # pixels

# Output file columns
columns = [
    "fname_RNA",
    "fname_condensate",
    "RNA_trackID",
    "t",
    "x",
    "y",
    "InCondensate",
    "condensateID",
    "R_nm",
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


def find_common(condensate_files, rna_files):
    experiment_names1 = [
        file.lstrip("condensates_AIO-").rstrip("-cropped-left.csv")
        for file in condensate_files
    ]
    experiment_names2 = [
        file.lstrip("SPT_results_AIO-").rstrip("-right_reformatted.csv")
        for file in rna_files
    ]
    return list(set(experiment_names1) & set(experiment_names2))


def fetch_nearby_condensates(
    df_condensate, t, mean_RNA_x, mean_RNA_y, interaction_cutoff
):
    # load condensates near the RNA as dictionary of polygons
    df_condensate_current_t = df_condensate[df_condensate["frame"] == t]

    # Calculate the squared distance between condensate centers and mean RNA coordinates
    df_condensate_current_t["distance_squared"] = (
        df_condensate_current_t["center_x_pxl"] - mean_RNA_x
    ) ** 2 + (df_condensate_current_t["center_y_pxl"] - mean_RNA_y) ** 2

    # Filter condensates within the interaction cutoff
    df_condensate_nearby = df_condensate_current_t[
        df_condensate_current_t["distance_squared"] <= interaction_cutoff
    ]

    # Create a dictionary to store nearby condensate polygons
    dict_condensate_polygons_nearby = {}

    # Iterate over nearby condensates and create polygons
    for _, row in df_condensate_nearby.iterrows():
        condensateID_nearby = row["condensateID"]
        str_condensate_coords = row["contour_coord"]

        lst_tup_condensate_coords = [
            tuple(map(int, coord.split(", ")))
            for coord in str_condensate_coords[2:-2].split("], [")
        ]

        dict_condensate_polygons_nearby[condensateID_nearby] = Polygon(
            lst_tup_condensate_coords
        )

    return dict_condensate_polygons_nearby


# matching files
rna_files = [
    file
    for file in os.listdir(join(folder_path, "RNA"))
    if file.startswith("SPT_results_AIO")
]
condensate_files = [
    file
    for file in os.listdir(join(folder_path, "condensate"))
    if file.startswith("condensates_AIO")
]
experiment_names = find_common(condensate_files, rna_files)


## loop through each FOV
for exp in track(experiment_names):
    rna_file = "SPT_results_AIO-" + exp + "-right_reformatted.csv"
    condensate_file = "condensates_AIO-" + exp + "-cropped-left.csv"
    cell_roi_files = [
        file
        for file in os.listdir("cell_body_mannual")
        if file.startswith(exp) and file.endswith(".txt")
    ]

    df_RNA = pd.read_csv(join(folder_path, "RNA", rna_file))
    df_condensate = pd.read_csv(join(folder_path, "condensate", condensate_file))

    ## process RNA tracks one by one
    lst_rows_of_df = []
    for trackID in df_RNA["trackID"]:
        current_track = df_RNA[df_RNA["trackID"] == trackID]
        lst_x = list_like_string_to_xyt(current_track["list_of_x"].squeeze())
        lst_y = list_like_string_to_xyt(current_track["list_of_y"].squeeze())
        lst_t = list_like_string_to_xyt(current_track["list_of_t"].squeeze())
        mean_RNA_x = np.mean(lst_x)
        mean_RNA_y = np.mean(lst_y)

        # process each position in track one by one
        for i in range(len(lst_t)):
            t = lst_t[i]
            x = lst_x[i]
            y = lst_y[i]

            point_RNA = Point(x, y)

            ## Perform colocalization
            # fetch nearby condensates
            dict_condensate_polygons_nearby = fetch_nearby_condensates(
                df_condensate, t, mean_RNA_x, mean_RNA_y, interaction_cutoff
            )
            # search for which condensate it's in
            InCondensate = False
            for key, polygon in dict_condensate_polygons_nearby.items():
                if point_RNA.within(polygon):
                    InCondensate = True
                    condensateID = key
                    R_nm = np.sqrt(polygon.area * nm_per_pixel**2 / np.pi)
                    # p1, p2 = nearest_points(polygon, point_RNA)
                    distance_to_edge_nm = (
                        polygon.exterior.distance(point_RNA) * nm_per_pixel
                    )
                    distance_to_center_nm = (
                        distance(polygon.centroid, point_RNA) * nm_per_pixel
                    )
                    break
            if not InCondensate:
                condensateID = np.nan
                R_nm = np.nan
                distance_to_center_nm = np.nan
                distance_to_edge_nm = np.nan

            # Save
            new_row = [
                rna_file,
                condensate_file,
                trackID,
                t,
                x,
                y,
                InCondensate,
                condensateID,
                R_nm,
                distance_to_center_nm,
                distance_to_edge_nm,
            ]
            lst_rows_of_df.append(new_row)

    df_save = pd.DataFrame.from_records(
        lst_rows_of_df,
        columns=columns,
    )
    fname_save = join(
        folder_path,
        "colocalization_AIO-" + exp + ".csv",
    )
    df_save.to_csv(fname_save, index=False)
