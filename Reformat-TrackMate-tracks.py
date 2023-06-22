import pandas as pd
import numpy as np
from os.path import dirname, basename, join
from tkinter import filedialog as fd
from rich.progress import track
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(color_codes=True, style="white")


def quick_tracklength_checker(df_formatted):
    df = df_formatted.sort_values(by=["trackID", "t"])
    t = df.t
    diff = np.append(0, np.ediff1d(t))
    # search for the starts of each track
    lst_idx_start = np.where(diff != 1)[0]
    # search for the ends of each track
    lst_idx_end = lst_idx_start[1:] - 1
    lst_idx_end = np.append(lst_idx_end, t.size - 1)  # padding the end
    # calculate track length
    tracklengths = lst_idx_end - lst_idx_start + 1

    return tracklengths


print("Please choose the csv file(s) from TrackMate exported tracks:")
lst_files = list(fd.askopenfilenames())

for fpath in track(lst_files):
    df_in = pd.read_csv(
        fpath,
        skiprows=[1, 2, 3],
        usecols=[
            "ID",
            "TRACK_ID",
            "QUALITY",
            "POSITION_X",
            "POSITION_Y",
            "POSITION_T",
            "RADIUS",
            "MEAN_INTENSITY_CH1",
            "MEDIAN_INTENSITY_CH1",
            "MIN_INTENSITY_CH1",
            "MAX_INTENSITY_CH1",
            "TOTAL_INTENSITY_CH1",
            "STD_INTENSITY_CH1",
            "CONTRAST_CH1",
            "SNR_CH1",
        ],
        dtype=float,
    )
    mapper = {
        "ID": "spotID",
        "TRACK_ID": "trackID",
        "POSITION_X": "x",
        "POSITION_Y": "y",
        "POSITION_T": "t",
        "RADIUS": "R",
        "MEAN_INTENSITY_CH1": "meanIntensity",
        "MEDIAN_INTENSITY_CH1": "medianIntensity",
        "MIN_INTENSITY_CH1": "minIntensity",
        "MAX_INTENSITY_CH1": "maxIntensity",
        "TOTAL_INTENSITY_CH1": "totalIntensity",
        "STD_INTENSITY_CH1": "stdIntensity",
        "CONTRAST_CH1": "contrast",
        "SNR_CH1": "SNR",
    }
    df_in = df_in.rename(columns=mapper)
    df_out = df_in.sort_values(by=["trackID", "t"])
    fpath_save = fpath[:-4] + "_reformatted.csv"
    df_out.to_csv(fpath_save, index=False)

    tracklengths = quick_tracklength_checker(df_out)
    plt.figure(figsize=(9, 4), dpi=200)
    sns.histplot(data=tracklengths, stat="count", bins=50, color="firebrick", alpha=0.5)
    plt.xlim(np.min(tracklengths), np.max(tracklengths))
    plt.title("Tracklength Distribution,\n" + basename(fpath), weight="bold")
    plt.xlabel("Number of Frames", weight="bold")
    plt.tight_layout()
    fpath_save = fpath[:-4] + "_TracklengthDist.png"
    plt.savefig(fpath_save, format="png")
    plt.close()
