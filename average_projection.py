from tkinter import filedialog as fd
import numpy as np
from tifffile import imread, imwrite
from rich.progress import track


lst_tifs = list(fd.askopenfilenames())

for fpath in track(lst_tifs):
    # perform average intensity projection, so any moving objects would blur out compared to maximum intensity projection
    img_average = np.mean(imread(fpath), axis=0).astype("uint16")
    # saving
    fname_save = fpath[:-4] + "_AveProj.tif"
    imwrite(fname_save, img_average, imagej=True)
