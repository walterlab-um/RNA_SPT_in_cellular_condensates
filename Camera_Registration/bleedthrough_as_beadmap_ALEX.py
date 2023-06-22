from os.path import join, dirname, basename
from tifffile import imread, imwrite
from tkinter import filedialog as fd
import numpy as np
import cv2
import pickle

fname = fd.askopenfilename()
video = imread(fname)
frames = int(video.shape[0])
frames_odd = np.arange(0, frames, 2)
video_even = np.delete(video, frames_odd, 0)
halfwidth = int(video.shape[2] / 2)
img_left = np.mean(video_even[:, :, 0:halfwidth], axis=0).astype("uint16")
img_right = np.mean(video_even[:, :, halfwidth:], axis=0).astype("uint16")
fsave_left = fname[:-4] + "-488excite-average-left.tif"
imwrite(fsave_left, img_left, imagej=True)
fsave_right = fname[:-4] + "-488excite-average-right.tif"
imwrite(fsave_right, img_right, imagej=True)


img_left = cv2.imread(fsave_left, cv2.IMREAD_GRAYSCALE)
img_right = cv2.imread(fsave_right, cv2.IMREAD_GRAYSCALE)
sz = img_left.shape
warp_mode = cv2.MOTION_HOMOGRAPHY
warp_matrix = np.eye(3, 3, dtype=np.float32)
number_of_iterations = 100

# Specify the threshold of the increment
# in the correlation coefficient between two iterations
termination_eps = 1e-5

# Define termination criteria
criteria = (
    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
    number_of_iterations,
    termination_eps,
)

# Run the ECC algorithm. The results are stored in warp_matrix.
(cc, warp_matrix) = cv2.findTransformECC(
    img_left, img_right, warp_matrix, warp_mode, criteria
)

# Save the matrix
folderpath = dirname(fname)
pickle.dump(warp_matrix, open(join(folderpath, "ONI_warp_matrix.p"), "wb"))

# Test and save alignment results on beads
img_right_unit16 = cv2.imread(fsave_right, -1)
img_right_aligned = cv2.warpPerspective(
    img_right_unit16,
    warp_matrix,
    (sz[1], sz[0]),
    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
)
imwrite(fsave_right[:-4] + "-aligned.tif", img_right_aligned, imagej=True)
