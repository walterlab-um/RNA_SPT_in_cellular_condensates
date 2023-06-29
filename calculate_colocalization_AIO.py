import os
from os.path import join, dirname, basename
import scipy.stats as stats
import numpy as np
import pandas as pd
from tkinter import filedialog as fd
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
angle_bins = np.linspace(0, 180, 7).astype(int)  # #boundaries = #bins + 1
lst_angle_fraction_titles = [
    "(" + str(angle_bins[i]) + "," + str(angle_bins[i + 1]) + "]"
    for i in range(len(angle_bins) - 1)
]
columns = [
    "filename",
    "trackID",
    "t",
    "x",
    "y",
    "cellID"，
    "InCondensate",
    "condensateID",
    "distance_to_center_nm",
    "distance_to_edge_nm",
    "N_steps",
    "Displacement_um",
    "mean_spot_intensity_max_in_track",
    "linear_fit_slope",
    "linear_fit_R2",
    "linear_fit_sigma",
    "linear_fit_D_um2s",
    "linear_fit_log10D",
    "loglog_fit_R2",
    "loglog_fit_log10D",
    "alpha",
    "list_of_angles",
] + +lst_angle_fraction_titles


lst_rows_of_df = []
print("Now Processing:", dirname(lst_fpath[0]))
for fpath in track(lst_fpath):
    df_current_file = pd.read_csv(fpath, dtype=float)
    df_current_file = df_current_file.astype({"t": int})
    fname = basename(fpath)
    lst_trackID_in_file = df_current_file.trackID.unique().tolist()

    for trackID in lst_trackID_in_file:
        df_current_track = df_current_file[df_current_file.trackID == trackID]
        tracklength = df_current_track.shape[0]
        if tracklength < tracklength_threshold:
            continue

        x = df_current_track["x"].to_numpy(dtype=float)
        y = df_current_track["y"].to_numpy(dtype=float)
        disp_um = np.sqrt((x[-1] - x[0]) ** 2 + (y[-1] - y[0]) ** 2) * um_per_pixel

        lags = np.arange(1, tracklength - 2)
        lags_phys = lags * s_per_frame
        MSDs = calc_MSD_NonPhysUnit(df_current_track, lags)
        if np.any(MSDs == 0):  # remove any track with zero MSD
            continue
        MSDs_phys = MSDs * (um_per_pixel**2)  # um^2

        # D formula with errors (MSD: um^2, t: s, D: um^2/s, n: dimension, R: motion blur coefficient; doi:10.1103/PhysRevE.85.061916)
        # diffusion dimension = 2. Note: This is the dimension of the measured data, not the actual movements! Although particles are doing 3D diffussion, the microscopy data is a projection on 2D plane and thus should be treated as 2D diffusion!
        # MSD = 2 n D tau + 2 n sigma^2 - 4 n R D tau, n=2, R=1/6
        # Therefore, slope = (2n-4nR)D = (8/3) D; intercept = 2 n sigma^2
        slope_linear, intercept_linear, R_linear, P, std_err = stats.linregress(
            lags_linear * s_per_frame, MSDs_phys[: len(lags_linear)]
        )
        if (slope_linear > 0) & (intercept_linear > 0):
            D_phys_linear = slope_linear / (8 / 3)  # um^2/s
            log10D_linear = np.log10(D_phys_linear)
            sigma_phys = np.sqrt(intercept_linear / 4) * 1000  # nm
        else:
            D_phys_linear = np.NaN
            log10D_linear = np.NaN
            sigma_phys = np.NaN

        # MSD = 2 n D tau^alpha = 4 D tau^alpha
        # log(MSD) = alpha * log(tau) + log(D) + log(4)
        # Therefore, slope = alpha; intercept = log(D) + log(4)
        slope_loglog, intercept_loglog, R_loglog, P, std_err = stats.linregress(
            np.log10(lags_loglog * s_per_frame),
            np.log10(MSDs_phys[: len(lags_loglog)]),
        )
        log10D_loglog = intercept_loglog - np.log10(4)
        alpha = slope_loglog

        angles = calc_angle(x, y)
        densities, _ = np.histogram(np.absolute(angles), angle_bins, density=True)
        # fractions are summed to 1; fraction = density * bin width
        fractions = densities * (angle_bins[1] - angle_bins[0])

        # Save
        new_row = [
            fname,
            trackID,
            df_current_track["t"].to_list(),
            df_current_track["x"].to_list(),
            df_current_track["y"].to_list(),
            tracklength,
            disp_um,
            x.mean(),
            y.mean(),
            df_current_track["meanIntensity"].max(),  # max of mean spot intensity
            MSDs_phys.tolist(),
            lags_phys.tolist(),
            slope_linear,
            R_linear**2,
            sigma_phys,
            D_phys_linear,
            log10D_linear,
            R_loglog**2,
            log10D_loglog,
            alpha,
            angles.tolist(),
        ] + fractions.tolist()
        lst_rows_of_df.append(new_row)

df_save = pd.DataFrame.from_records(
    lst_rows_of_df,
    columns=columns,
)
fname_save = join(dirname(fpath), "SPT_results_AIO-pleaserename.csv")
df_save.to_csv(fname_save, index=False)
