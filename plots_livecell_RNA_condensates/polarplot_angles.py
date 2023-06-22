import os, sys
import pandas as pd
import numpy as np
from progress.bar import IncrementalBar
from tkinter import filedialog as fd


##############################
# Settings
print("Select the folder to all *RNA* *TRACKS*:")
folderpath_RNA = fd.askdirectory()
print("The RNA folder is: ", folderpath_RNA)

print('Please type below exactly "ONI" or "SMART" for microscpoe type:')
microscope = input()  # either 'ONI' or 'SMART'

# folderpath_RNA = '/Volumes/AnalysisGG/KNIME_results_from_greatlakes/20211007-RNAs'
# min_disp = 10
# microscope = 'ONI'

##############################
# Function
def calc_angle(x, y):
    # x and y at time 0 and time 1
    x0 = x[:-1]
    x1 = x[1:]
    y0 = y[:-1]
    y1 = y[1:]
    # unit vectors of all steps, and step 0 and step 1
    vector = np.array([x1 - x0, y1 - y0])
    # convert to complex number to use np.angle
    complex = 1j * vector[1, :]
    complex += vector[0, :]
    angles_eachstep = np.angle(complex, deg=True)
    angles = np.ediff1d(angles_eachstep)  # between adjacent steps
    # convert all angles to within range (0,+-180) for output
    angles[angles < -180] = angles[angles < -180] + 360
    angles[angles > 180] = angles[angles > 180] - 360

    return angles


##############################
# Main
if microscope == "SMART":
    RNA_side = "_left"
    um_per_pixel = 0.134
elif microscope == "ONI":
    RNA_side = "_right"
    um_per_pixel = 0.117
else:
    print("The variable microscpe must be either SMART or ONI!")
    sys.exit()


lst_fnames_RNA = [f for f in os.listdir(folderpath_RNA) if f.endswith("_filtered.csv")]

os.chdir(folderpath_RNA)
bar = IncrementalBar(
    "Processing...", suffix="%(percent).1f%% - %(eta)ds", max=len(lst_fnames_RNA)
)
fname = lst_fnames_RNA[0]
for fname in lst_fnames_RNA:
    df_RNA = pd.read_csv(fname, dtype=float)

    lst_x0 = np.array([], dtype=float)
    lst_y0 = np.array([], dtype=float)
    lst_disp = np.array([], dtype=float)
    lst_disp_for_angles = np.array([], dtype=float)
    lst_angles = np.array([], dtype=float)

    for trackID in df_RNA.trackID.unique():
        x = df_RNA[df_RNA.trackID == trackID].x.to_numpy(dtype=float)
        y = df_RNA[df_RNA.trackID == trackID].y.to_numpy(dtype=float)
        disp = np.sqrt((x[-1] - x[0]) ** 2 + (y[-1] - y[0]) ** 2)

        lst_x0 = np.append(lst_x0, x[0])
        lst_y0 = np.append(lst_y0, y[0])
        lst_disp = np.append(lst_disp, disp)

        angles = calc_angle(x, y)
        lst_angles = np.append(lst_angles, angles)
        disp_repeats = np.repeat(disp, angles.size)
        lst_disp_for_angles = np.append(lst_disp_for_angles, disp_repeats)

    df_save1 = pd.DataFrame(dtype=float)
    df_save1["initial_x"] = lst_x0
    df_save1["initial_y"] = lst_y0
    df_save1["displacement_um"] = lst_disp * um_per_pixel
    fname_save = fname.split(RNA_side)[0] + "_totaldisp.csv"
    df_save1.to_csv(fname_save, index=False)

    df_save2 = pd.DataFrame(dtype=float)
    df_save2["angles"] = lst_angles
    df_save2["displacement_um"] = lst_disp_for_angles * um_per_pixel
    fname_save = fname.split(RNA_side)[0] + "_angles.csv"
    df_save2.to_csv(fname_save, index=False)

    bar.next()
bar.finish()
