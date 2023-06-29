from os.path import join, dirname, basename
import scipy.stats as stats
import numpy as np
import pandas as pd
from tkinter import filedialog as fd
from rich.progress import track

pd.options.mode.chained_assignment = None  # default='warn'

# AIO: All in one format


# scalling factors for physical units
print("Default scaling factors: s_per_frame = 0.02, um_per_pixel = 0.117")
# print("micron per pixel:")
# um_per_pixel = float(input())
um_per_pixel = 0.117
# print("Enter the time between frames (seconds):")
# s_per_frame = float(input())
s_per_frame = 0.1

print("Choose the RNA-bandpass_reformatted.csv files for processing:")
lst_fpath = list(fd.askopenfilenames())
tracklength_threshold = 8  # must be 3 + max{len(lags_linear), len(lags_loglog)}
# lag times for linear MSD-tau fitting (unit: frame)
lags_linear = np.arange(1, 6, 1)
# lag times for log-log MSD-tau fitting (unit: frame)
lags_loglog = np.arange(1, 6, 1)

# Output file columns
angle_bins = np.linspace(0, 180, 7).astype(int)  # #boundaries = #bins + 1
lst_angle_fraction_titles = [
    "(" + str(angle_bins[i]) + "," + str(angle_bins[i + 1]) + "]"
    for i in range(len(angle_bins) - 1)
]
columns = [
    "filename",
    "trackID",
    "list_of_t",
    "list_of_x",
    "list_of_y",
    "N_steps",
    "Displacement_um",
    "mean_x_pxl",
    "mean_y_pxl",
    "mean_spot_intensity_max_in_track",
    "list_of_MSD_um2",
    "list_of_tau_s",
    "linear_fit_slope",
    "linear_fit_R2",
    "linear_fit_sigma",
    "linear_fit_D_um2s",
    "linear_fit_log10D",
    "loglog_fit_R2",
    "loglog_fit_log10D",
    "alpha",
    "list_of_angles",
] + lst_angle_fraction_titles


def calc_MSD_NonPhysUnit(df_track, lags):
    df_track_sorted = df_track.sort_values("t")
    # prepare storage arrays
    MSDs = []
    # STDs = np.array([], dtype=float)

    # filling gaps
    ref_t = list(range(df_track_sorted.t.min(), df_track_sorted.t.max()))
    missing_t = list(set(ref_t) - set(df_track_sorted.t))
    if len(missing_t) > 0:
        for t in missing_t:
            complete_t = np.append(df_track_sorted.t, np.array(missing_t, dtype=int))
            complete_x = np.append(df_track_sorted.x, np.repeat(np.NaN, len(missing_t)))
            complete_y = np.append(df_track_sorted.y, np.repeat(np.NaN, len(missing_t)))
            df_complete = pd.DataFrame()
            df_complete["x"] = complete_x
            df_complete["y"] = complete_y
            df_complete["t"] = complete_t
            df_complete = df_complete.sort_values("t")

    else:
        df_complete = df_track_sorted

    # calculate MSDs corresponding to a series of lag times
    for lag in lags:
        Xs = np.array(df_complete.x, dtype=float)
        Ys = np.array(df_complete.y, dtype=float)

        SquareDisplacements = (Xs[lag:] - Xs[:-lag]) ** 2 + (Ys[lag:] - Ys[:-lag]) ** 2
        MSD = np.nanmean(SquareDisplacements)
        MSDs.append(MSD)

    MSDs = np.array(MSDs, dtype=float)
    return MSDs


def calc_angle(x, y):
    # x and y at time 0 and time 1
    x0 = x[:-1]
    x1 = x[1:]
    y0 = y[:-1]
    y1 = y[1:]
    # unit vectors of all steps, and step 0 and step 1
    vector = np.array([x1 - x0, y1 - y0])
    # convert to complex number to use np.angle
    vector_complex = 1j * vector[1, :]
    vector_complex += vector[0, :]
    angles_eachstep = np.angle(vector_complex, deg=True)
    angles = np.ediff1d(angles_eachstep)  # between adjacent steps
    # convert all angles to within range (0,+-180) for output
    angles[angles < -180] = angles[angles < -180] + 360
    angles[angles > 180] = angles[angles > 180] - 360

    return angles


print("Now Processing:", dirname(lst_fpath[0]))
for fpath in track(lst_fpath):
    df_current_file = pd.read_csv(fpath, dtype=float)
    df_current_file = df_current_file.astype({"t": int})
    fname = basename(fpath)
    lst_trackID_in_file = df_current_file.trackID.unique().tolist()

    lst_rows_of_df = []
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
    fname_save = join(dirname(fpath), "SPT_results_AIO-" + basename(fpath))
    df_save.to_csv(fname_save, index=False)
