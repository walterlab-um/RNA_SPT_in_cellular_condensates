import os
from os.path import join, dirname
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points
from shapely import distance
from rich.progress import track

pd.options.mode.chained_assignment = None  # default='warn'

# AIO: All in one format
# This script gives the collocalization information for every RNA position over time.
# This version is for live cell data, including cell boundaries


# scalling factors for physical units
nm_per_pixel = 117
print("Scaling factors: nm_per_pixel = " + str(nm_per_pixel))

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

        # load cell boundaries as dictionary of polygons
        fnames_cells_in_current_FOV = []
        dict_cell_polygons = dict()
        for f in all_fname_cell_body:
            if fname_RNA_AIO[16:].split("-RNA")[0] + "-" in f:
                fnames_cells_in_current_FOV.append(f)
                cell_outline_coordinates = pd.read_csv(
                    join(subfolder, "cell_body", f), sep="	", header=None
                )
                coords_roi = [
                    tuple(row) for _, row in cell_outline_coordinates.iterrows()
                ]
                dict_cell_polygons[f] = Polygon(coords_roi)
                break
        if len(fnames_cells_in_current_FOV) == 0:
            print("There's no cells in current FOV.")
            continue

        # load all condensates within the current FOV
        for f in all_fname_condensate_AIO:
            if fname_RNA_AIO[16:].split("-RNA")[0] + "-" in f:
                fname_condensate_AIO = f
                break
        if "fname_condensate_AIO" not in globals():
            print("There's no condensate AIO file for the current FOV.")
            continue
        df_condensate = pd.read_csv(join(subfolder, "condensate", fname_condensate_AIO))

        ## process RNA tracks one by one
        lst_rows_of_df = []
        for trackID in track(df_RNA["trackID"], description=fname_RNA_AIO):
            current_track = df_RNA[df_RNA["trackID"] == trackID]
            lst_x = list_like_string_to_xyt(current_track["list_of_x"].squeeze())
            lst_y = list_like_string_to_xyt(current_track["list_of_y"].squeeze())
            lst_t = list_like_string_to_xyt(current_track["list_of_t"].squeeze())
            mean_RNA_x = np.mean(lst_x)
            mean_RNA_y = np.mean(lst_y)

            # load condensates near the RNA as dictionary of polygons
            df_condensate_current_t = df_condensate[df_condensate["frame"] == t]
            all_condensateID_nearby = []
            for _, row in df_condensate_current_t.iterrows():
                center_x_pxl = row["center_x_pxl"]
                center_y_pxl = row["center_y_pxl"]
                if (center_x_pxl - mean_RNA_x) ** 2 + (
                    center_y_pxl - mean_RNA_y
                ) ** 2 > 50:
                    continue
                else:
                    all_condensateID_nearby.append(row["condensateID"])

            dict_condensate_polygons_nearby = dict()
            for condensateID_nearby in all_condensateID_nearby:
                str_condensate_coords = df_condensate_current_t[
                    df_condensate_current_t["condensateID"] == condensateID_nearby
                ]["contour_coord"].squeeze()

                lst_tup_condensate_coords = []
                for str_condensate_xy in str_condensate_coords[2:-2].split("], ["):
                    condensate_x, condensate_y = str_condensate_xy.split(", ")
                    lst_tup_condensate_coords.append(
                        tuple([int(condensate_x), int(condensate_y)])
                    )

                dict_condensate_polygons_nearby[condensateID_nearby] = Polygon(
                    lst_tup_condensate_coords
                )

            # process each position in track one by one
            for i in range(len(lst_t)):
                t = lst_t[i]
                x = lst_x[i]
                y = lst_y[i]

                point_RNA = Point(x, y)
                ## Perform colocalization
                # search for which cell it's in
                for key, polygon in dict_cell_polygons.items():
                    if point_RNA.within(polygon):
                        fname_cell = key
                        break
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
                    fname_RNA_AIO,
                    fname_condensate_AIO,
                    fname_cell,
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
            subfolder,
            "colocalization_AIO-"
            + fname_RNA_AIO.split("AIO-")[-1].split("-RNA")[0]
            + ".csv",
        )
        df_save.to_csv(fname_save, index=False)
