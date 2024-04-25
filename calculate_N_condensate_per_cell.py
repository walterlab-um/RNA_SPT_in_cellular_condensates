from shapely.geometry import Point, Polygon
import os
from os.path import join, dirname
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from tkinter import filedialog as fd
import tkinter as tk

pd.options.mode.chained_assignment = None  # default='warn'


def transform_string(input_string):
    # Split the input string by '-' to get the parts
    parts = input_string.split("-")

    # Remove the last part (e.g., 'cell1.txt')
    parts.pop()

    # Join the remaining parts back together with '-'
    output_string = "-".join(parts)

    # Add the prefix and postfix to the output string
    output_string = "condensates_AIO-" + output_string + ".csv"

    return output_string


def count_within_cell_polygon(df, cell_polygon):
    """
    Take a Polygon cell_polygon and a dataframe contain columns 'x' and 'y', and return numpy array of x and y within the cell_polygon.
    """
    count = 0
    for _, row in df.iterrows():
        if Point(row.x, row.y).within(cell_polygon):
            count += 1
    return count


# Function to process a single file
def process_file(
    i,
    condensate_files,
    cell_roi_files,
    folder_path,
):
    # import cell boundary as a polygon
    cell_roi_file = cell_roi_files[i]
    cell_outline_coordinates = pd.read_csv(
        join(folder_path, "cell_body_mannual", cell_roi_file), sep="	", header=None
    )
    coords_roi = [tuple(row) for _, row in cell_outline_coordinates.iterrows()]
    cell_polygon = Polygon(coords_roi)

    # import condensates
    df_condensate = pd.read_csv(
        join(folder_path, "condensate", transform_string(cell_roi_file))
    )

    N_condensate_per_cell = 0
    for _, row in df_condensate.iterrows():
        if Point(row.x, row.y).within(cell_polygon):
            N_condensate_per_cell += 1

    return (
        cell_roi_file,
        N_condensate_per_cell,
    )


def main():
    print(
        "Choose the main folder contains 3 subfolders: immuno_protein, condensate, cell_body_mannual"
    )
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    folder_path = fd.askdirectory()
    fname_save = "N_condensate_per_cell.csv"

    cell_roi_files = [
        file
        for file in os.listdir(join(folder_path, "cell_body_mannual"))
        if file.endswith(".txt")
    ]
    condensate_files = [transform_string(f) for f in cell_roi_files]

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
                    condensate_files,
                    cell_roi_files,
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
        lst_N_condensate_per_cell,
    ) = map(list, zip(*processed_results))

    df_save = pd.DataFrame(
        {"cell_rois": lst_cell_roi, "N_condensate_per_cell": lst_N_condensate_per_cell}
    )
    df_save.to_csv(join(folder_path, fname_save), index=False)
    print("Saved successfully at the following path:")
    print(join(folder_path, fname_save))


if __name__ == "__main__":
    main()
