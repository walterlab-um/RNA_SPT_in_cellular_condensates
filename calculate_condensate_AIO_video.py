from tifffile import imread
import os
from os.path import join, dirname, basename
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tkinter import filedialog as fd
from rich.progress import track

pd.options.mode.chained_assignment = None  # default='warn'

# AIO: All in one format


um_per_pixel = 0.117
print("Choose the Simple Segmentation.tif files for processing:")
lst_fpath = list(fd.askopenfilenames())
folder_save = dirname(lst_fpath[0])
os.chdir(folder_save)

# output data structure
columns = [
    "filename",
    "condensateID",
    "contour_coord",
    "center_x_pxl",
    "center_y_pxl",
    "area_um2",
    "R_nm",
    "mean_intensity",
    "max_intensity",
    "max_location",
    "aspect_ratio",
    "contour_solidity",
    "contour_extent",
]

switch_plot = True  # a switch to turn off plotting
plow = 0.05  # imshow intensity percentile
phigh = 98


####################################
# Functions
def pltcontours(img, contours, fsave):
    global rescale_contrast, plow, phigh
    plt.figure(dpi=600)
    # Contrast stretching
    vmin, vmax = np.percentile(img, (plow, phigh))
    plt.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
    for cnt in contours:
        x = cnt[:, 0][:, 0]
        y = cnt[:, 0][:, 1]
        plt.plot(x, y, "-", color="firebrick", linewidth=0.2)
        # still the last closing line will be missing, get it below
        xlast = [x[-1], x[0]]
        ylast = [y[-1], y[0]]
        plt.plot(xlast, ylast, "-", color="firebrick", linewidth=0.2)
    plt.xlim(0, img.shape[0])
    plt.ylim(0, img.shape[1])
    plt.tight_layout()
    plt.axis("scaled")
    plt.axis("off")
    plt.savefig(fsave, format="png", bbox_inches="tight", dpi=600)
    plt.close()


def cnt_fill(imgshape, cnt):
    # create empty image
    mask = np.zeros(imgshape, dtype=np.uint8)
    # draw contour
    cv2.fillPoly(mask, [cnt], (255))

    return mask


def get_circular_kernel(diameter):
    mid = (diameter - 1) / 2
    distances = np.indices((diameter, diameter)) - np.array([mid, mid])[:, None, None]
    kernel = ((np.linalg.norm(distances, axis=0) - mid) <= 0).astype("uint8")

    return kernel


def cnt_to_list(cnt):
    # convert cv2's cnt format to list of corrdinates, for the ease of saving in one dataframe cell
    cnt_2d = np.reshape(cnt, (cnt.shape[0], cnt.shape[2]))
    lst_cnt = [cnt_2d[i, :].tolist() for i in range(cnt_2d.shape[0])]

    return lst_cnt


lst_rows_of_df = []
print("Now Processing:", dirname(lst_fpath[0]))
for fpath in track(lst_fpath):
    ilastik_output = imread(fpath)
    video = imread(fpath[:-24] + ".tif")
    filename = basename(fpath)[:-24]

    mask_all_condensates = 2 - ilastik_output  # background label=2, condensate label=1
    if mask_all_condensates.sum() == 0:  # so no condensate
        continue

    # find contours coordinates in binary edge image. contours here is a list of np.arrays containing all coordinates of each individual edge/contour.
    contours, _ = cv2.findContours(
        mask_all_condensates, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )
    condensateID = 1
    for cnt in contours:
        # ignore contour if it's as large as the FOV, becuase cv2 recognizes the whole image as a contour
        if cv2.contourArea(cnt) > 0.8 * video.shape[1] * video.shape[2]:
            continue

        # condensate center coordinates
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        center_x_pxl = int(M["m10"] / M["m00"])
        center_y_pxl = int(M["m01"] / M["m00"])
        # condensate size
        area_um2 = cv2.contourArea(cnt) * um_per_pixel**2
        R_nm = np.sqrt(area_um2 / np.pi) * 1000
        # aspect ratio
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h
        # extent
        rect_area = w * h
        contour_extent = float(cv2.contourArea(cnt)) / rect_area
        # solidity
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        contour_solidity = float(cv2.contourArea(cnt)) / hull_area
        # intensities
        mask_current_condensate = cnt_fill(img.shape, cnt)
        mean_intensity = cv2.mean(img, mask=mask_current_condensate)[0]
        _, max_intensity, _, max_location = cv2.minMaxLoc(
            img, mask=mask_current_condensate
        )

        # Finally, add the new row to the list to form dataframe
        new_row = [
            filename,
            condensateID,
            cnt_to_list(cnt),
            center_x_pxl,
            center_y_pxl,
            area_um2,
            R_nm,
            mean_intensity,
            max_intensity,
            max_location,
            aspect_ratio,
            contour_solidity,
            contour_extent,
        ]
        lst_rows_of_df.append(new_row)

        condensateID += 1

    if switch_plot:
        pltcontours(img, contours, fpath[:-4] + ".png")
    else:
        continue

df_save = pd.DataFrame.from_records(
    lst_rows_of_df,
    columns=columns,
)
fname_save = join(dirname(fpath), "condensates_AIO-pleaserename.csv")
df_save.to_csv(fname_save, index=False)
