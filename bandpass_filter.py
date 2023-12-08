from tifffile import imread, imwrite
from scipy.ndimage import white_tophat
from skimage.filters import difference_of_gaussians
from skimage.util import img_as_uint
from tkinter import filedialog as fd
import numpy as np
from rich.progress import track

print("Choose all tif files to be batch proccessed:")
lst_files = list(fd.askopenfilenames())

DoG_sigma_low = 1
DoG_sigma_high = 3

###############################
for f in track(lst_files):
    video = imread(f)
    video_out = []
    for img in video:
        img_filtered = difference_of_gaussians(img, DoG_sigma_low, DoG_sigma_high)
        video_out.append(img_filtered)
    video_out = np.stack(video_out)
    fsave = f[:-4] + "-bandpass.tif"
    if len(video.shape) < 3:  # image:
        imwrite(fsave, img_as_uint(video_out), imagej=True)
    elif len(video.shape) > 2:  # video:
        imwrite(fsave, img_as_uint(video_out), imagej=True, metadata={"axes": "TYX"})
