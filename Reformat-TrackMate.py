import pandas as pd
from tkinter import filedialog as fd
from rich.progress import track


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
