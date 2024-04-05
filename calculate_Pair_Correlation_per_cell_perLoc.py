from shapely.geometry import Point, Polygon
from scipy.ndimage import gaussian_filter
from tifffile import imread
import cv2
import math
import os
from os.path import join, dirname
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from tkinter import filedialog as fd
import tkinter as tk

pd.options.mode.chained_assignment = None  # default='warn'


"""
The program calculates per-location Pair Correlation Function (PCF; aka. Radial Distribution Function, RDF) of two channel particles (condensates and smiFISH/Immuno-fluo puncta).
It saves four PCF with corresponding informatioon in a single pickle file:
cross PCF using channel 1 spots as reference
"""


def find_common(condensate_files, rna_files):
    experiment_names1 = [file.rstrip("-ch1.csv") for file in condensate_files]
    experiment_names2 = [file.rstrip("-ch2.csv") for file in rna_files]
    return list(set(experiment_names1) & set(experiment_names2))


def corr_within_cell_polygon(df, cell_polygon):
    """
    Take a Polygon cell_polygon and a dataframe contain columns 'x' and 'y', and return numpy array of x and y within the cell_polygon.
    """
    lst_x = []
    lst_y = []
    for _, row in df.iterrows():
        if Point(row.x, row.y).within(cell_polygon):
            lst_x.append(row.x)
            lst_y.append(row.y)
    array_x = np.array(lst_x, dtype=float)
    array_y = np.array(lst_y, dtype=float)
    return array_x, array_y


def PairCorr_with_edge_correction(
    df_ref,
    df_interest,
    cell_polygon,
    nm_per_pxl,
    r_max_nm,
    ringwidth_nm,
    dr_slidingrings_nm,
):
    # only count particles within cell_polygon
    x_ref, y_ref = corr_within_cell_polygon(df_ref, cell_polygon)
    x_interest, y_interest = corr_within_cell_polygon(df_interest, cell_polygon)

    # Total number particles in cell_polygon
    N_ref = x_ref.shape[0]
    N_interest = x_interest.shape[0]

    # particle density rho, unit: number per nano meter square
    cell_polygon_area_nm2 = cell_polygon.area * (nm_per_pxl**2)
    rho_ref_per_nm2 = N_ref / cell_polygon_area_nm2
    rho_interest_per_nm2 = N_interest / cell_polygon_area_nm2

    # setup bins and ring areas
    bin_starts = np.arange(0, r_max_nm - ringwidth_nm, dr_slidingrings_nm)
    bin_ends = bin_starts + ringwidth_nm
    ring_areas_nm2 = np.pi * (
        bin_ends**2 - bin_starts**2
    )  # area of rings, unit nm square
    ring_areas_pxl2 = ring_areas_nm2 / (nm_per_pxl**2)

    # Calculate corrected histogram of distances
    lst_hist_per_point_cross = []
    for i in range(len(x_ref)):
        # Calculate edge correction factor
        rings = [
            Point(x_ref[i], y_ref[i])
            .buffer(end)
            .difference(Point(x_ref[i], y_ref[i]).buffer(start))
            for start, end in zip(bin_starts / nm_per_pxl, bin_ends / nm_per_pxl)
        ]
        intersect_areas = np.array(
            [
                cell_polygon.intersection(Polygon(ring), grid_size=0.1).area
                for ring in rings
            ]
        )
        edge_correction_factors = 1 / (intersect_areas / ring_areas_pxl2)

        # cross correlation
        lst_hist = []
        for j in range(len(x_interest)):
            distance = (
                np.sqrt(
                    (x_ref[i] - x_interest[j]) ** 2 + (y_ref[i] - y_interest[j]) ** 2
                )
                * nm_per_pxl
            )
            lst_hist.append(((bin_starts <= distance) & (bin_ends >= distance)) * 1)
        hist_per_point_corrected = np.sum(lst_hist, axis=0) * edge_correction_factors
        lst_hist_per_point_cross.append(hist_per_point_corrected)

    # calculate normalization factor that counts for density and ring area
    norm_factors_cross = N_ref * ring_areas_nm2 * rho_interest_per_nm2

    PairCorr_cross = np.sum(lst_hist_per_point_cross, axis=0) / norm_factors_cross

    return PairCorr_cross


# Function to process a single file
def process_file(
    i,
    rna_files,
    condensate_files,
    cell_roi_files,
    nm_per_pxl,
    r_max_nm,
    ringwidth_nm,
    dr_slidingrings_nm,
    folder_path,
):
    # import cell boundary as a polygon
    cell_roi_file = cell_roi_files[i]
    cell_outline_coordinates = pd.read_csv(
        join(folder_path, "cell_body_mannual", cell_roi_file), sep="	", header=None
    )
    coords_roi = [tuple(row) for _, row in cell_outline_coordinates.iterrows()]
    cell_polygon = Polygon(coords_roi)

    # import condensates (ch1) and RNA/protein (ch2) spots
    matching_rna_file = [
        s for s in rna_files if s.startswith(cell_roi_file.split("-cell")[0])
    ][0]
    matching_condensate_file = [
        s for s in condensate_files if s.startswith(cell_roi_file.split("-cell")[0])
    ][0]
    df_rna = pd.read_csv(join(folder_path, "RNA", matching_rna_file))
    df_condensate = pd.read_csv(
        join(folder_path, "condensate", matching_condensate_file)
    )

    cross_condensate_ref = PairCorr_with_edge_correction(
        df_condensate,  # ref
        df_rna,  # interest
        cell_polygon,
        nm_per_pxl,
        r_max_nm,
        ringwidth_nm,
        dr_slidingrings_nm,
    )

    return (
        cell_roi_file,
        matching_rna_file,
        matching_condensate_file,
        cross_condensate_ref,
        df_condensate.shape[0],
        df_rna.shape[0],
    )


def main():
    print(
        "Choose the main folder contains 3 subfolders: RNA, condensate, cell_body_mannual"
    )
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    folder_path = fd.askdirectory()
    os.chdir(folder_path)
    fname_save = "PairCorr-DataDict-pooled-perLoc.p"

    # Parameters
    print("Please enter pixel size in nm:")
    nm_per_pxl = float(input())
    r_max_nm = 1120
    ringwidth_nm = 100
    dr_slidingrings_nm = 20  # stepsize between adjascent overlaping rings, nm
    bins = np.arange(
        0, r_max_nm - ringwidth_nm, dr_slidingrings_nm
    )  # overlaping bins (sliding window)

    # Matching three folder contents
    rna_files = [file for file in os.listdir("RNA") if file.endswith("-ch2.csv")]
    condensate_files = [
        file for file in os.listdir("condensate") if file.endswith("-ch1.csv")
    ]
    experiment_names = find_common(condensate_files, rna_files)
    cell_roi_files = [
        file
        for file in os.listdir("cell_body_mannual")
        if any(file.startswith(name) for name in experiment_names)
        and file.endswith(".txt")
    ]

    # The tqdm library provides an easy way to visualize the progress of loops.
    pbar = tqdm(total=len(cell_roi_files))

    # Update function for the progress bar
    def update(*a):
        pbar.update()

    # Create a process pool and map the function to the files
    with Pool(cpu_count()) as p:
        results = []
        for i in range(len(cell_roi_files)):
            result = p.apply_async(
                process_file,
                args=(
                    i,
                    rna_files,
                    condensate_files,
                    cell_roi_files,
                    nm_per_pxl,
                    r_max_nm,
                    ringwidth_nm,
                    dr_slidingrings_nm,
                    folder_path,
                ),
                callback=update,
            )
            results.append(result)

        processed_results = []
        for r in results:
            try:
                processed_results.append(r.get())
            except:
                pass
    pbar.close()

    # Unpack results into separate lists
    (
        lst_cell_roi,
        lst_rna_file,
        lst_condensate_file,
        lst_cross,
        lst_size_FUS,
        lst_size_RNA,
    ) = map(list, zip(*processed_results))

    dict_to_save = {
        "cell_rois": lst_cell_roi,
        "filenames_RNA": lst_rna_file,
        "filenames_condensate": lst_condensate_file,
        "lst_N_locations_FUS": lst_size_FUS,
        "lst_N_locations_RNA": lst_size_RNA,
        "lst_cross": lst_cross,
        "nm_per_pxl": nm_per_pxl,
        "r_max_nm": r_max_nm,
        "ringwidth_nm": ringwidth_nm,
        "dr_slidingrings_nm": dr_slidingrings_nm,
        "bins": bins,
    }
    pickle.dump(
        dict_to_save,
        open(join(folder_path, fname_save), "wb"),
    )
    print("Saved successfully at the following path:")
    print(join(folder_path, fname_save))


if __name__ == "__main__":
    main()
