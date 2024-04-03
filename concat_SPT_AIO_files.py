import os
from os.path import dirname, basename
import numpy as np
import pandas as pd
from rich.progress import track
from tkinter import filedialog as fd

print("Choose all SPT_results_AIO_xxxxx.csv files within the same condition:")
lst_path_data = list(fd.askopenfilenames())
folder_data = dirname(lst_path_data[0])
os.chdir(folder_data)
lst_df = []
for f in track(lst_path_data):
    df = pd.read_csv(f)
    df.insert(0, "filename", np.repeat(basename(f), df.shape[0]))
    lst_df.append(df)
df_all = pd.concat(lst_df, ignore_index=True)
df_all.to_csv("SPT_results_AIO_concat-pleaserename.csv", index=False)
