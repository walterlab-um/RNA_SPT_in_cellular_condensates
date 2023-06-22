import os
from os.path import join, dirname, basename
import cv2
from tifffile import imread, imwrite
import pickle
import numpy as np
from tkinter import filedialog as fd
from rich.progress import track
from rich import print as rprint
import shutil

#########################################
# Load and organize files
rprint("[red]Choose the matrix[red]")
path_matrix = fd.askopenfilename()

rprint(
    "path_matrix:",
    "[red]" + path_matrix + "[red]",
)

warp_matrix = pickle.load(open(path_matrix, "rb"))
rprint("[red]Choose all tif files to be processed[red]")
lst_files = list(fd.askopenfilenames())


def crop_imgstack(imgstack):
    # dimension of the imgstack should be (z,h,w)
    # this will crop a 5 pixel frame from the image
    z, h, w = imgstack.shape
    imgstack_out = imgstack[:, 5 : h - 5, 5 : w - 5]
    return imgstack_out


def transform(img2d, warp_matrix):
    sz = img2d.shape
    img_out = cv2.warpPerspective(
        img2d,
        warp_matrix,
        (sz[1], sz[0]),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
    )
    return img_out


for fpath in track(lst_files):
    # load the tiff file
    img = imread(fpath)
    halfwidth = int(img.shape[2] / 2)

    # split left and right
    img_left = img[:, :, 0:halfwidth]
    img_right = img[:, :, halfwidth:]

    # directly crop and save the left
    img_left_cropped = crop_imgstack(img_left)
    fsave_left = fpath.rstrip(".tif") + "-cropped-left.tif"
    imwrite(fsave_left, img_left_cropped, imagej=True, metadata={"axes": "TYX"})

    # Use warpPerspective for Homography transform ON EACH Z FRAME
    lst_img_right_aligned = [
        transform(img_right[z, :, :], warp_matrix) for z in range(img.shape[0])
    ]
    img_right_aligned = np.stack(lst_img_right_aligned, axis=0)
    img_right_cropped = crop_imgstack(img_right_aligned)
    fsave_right = fpath.rstrip(".tif") + "-cropped-right.tif"
    imwrite(
        fsave_right,
        img_right_cropped,
        imagej=True,
        metadata={"axes": "TYX"},
    )
