from tifffile import imread, imwrite
import cv2
import pickle
import numpy as np
from tkinter import filedialog as fd
from rich.progress import track


print("Choose the matrix")
path_matrix = fd.askopenfilename()

warp_matrix = pickle.load(open(path_matrix, "rb"))

print("Choose the tif files for channel alignment:")
lst_files = list(fd.askopenfilenames())

print("Type in 1 for odd left even right; 2 for odd right even left")
selector = int(input())


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


#########################################
# Apply registration and Crop
for fname in track(lst_files):
    # load the tiff file
    img = imread(fname)
    halfwidth = int(img.shape[2] / 2)

    # even and odd frame indexes
    frames = int(img.shape[0])
    frames_odd = np.arange(0, frames, 2)
    frames_even = np.arange(1, frames + 1, 2)

    # split left and right
    img_left = img[:, :, 0:halfwidth]
    img_right = img[:, :, halfwidth:]

    img_left_cropped = crop_imgstack(img_left)
    if selector == 1:  # odd left even right
        img_left_final = np.delete(img_left_cropped, frames_even, 0)
    elif selector == 2:  # odd right even left
        img_left_final = np.delete(img_left_cropped, frames_odd, 0)

    fsave_left = fname.strip(".tif") + "-left.tif"
    imwrite(
        fsave_left,
        img_left_final,
        imagej=True,
        metadata={"axes": "TYX"},
    )

    # Use warpPerspective for Homography transform ON EACH Z FRAME
    lst_img_right_aligned = [
        transform(img_right[z, :, :], warp_matrix) for z in range(img.shape[0])
    ]
    img_right_aligned = np.stack(lst_img_right_aligned, axis=0)

    # crop, discard the even for the right
    img_right_cropped = crop_imgstack(img_right_aligned)
    if selector == 1:  # odd left even right
        img_right_final = np.delete(img_right_cropped, frames_odd, 0)
    elif selector == 2:  # odd right even left
        img_right_final = np.delete(img_right_cropped, frames_even, 0)
    fsave_right = fname.strip(".tif") + "-right.tif"
    imwrite(
        fsave_right,
        img_right_final,
        imagej=True,
        metadata={"axes": "TYX"},
    )
