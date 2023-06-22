import os
import cv2
import pickle
from tifffile import imread, imwrite
from tkinter import filedialog as fd
import numpy as np

path = fd.askopenfilename(title="Choose a bead image:")
img = imread(path)
# For ONI:
img_left = img[:, 0:428]
img_right = img[:, 428:]
# For SMART center SPT:
# img_left = img[:, 0:512]
# img_right = img[:, 512:]
imwrite(path[:-4] + "-left.tif", img_left, imagej=True)
imwrite(path[:-4] + "-right.tif", img_right, imagej=True)


img_left = cv2.imread(path[:-4] + "-left.tif", cv2.IMREAD_GRAYSCALE)
img_right = cv2.imread(path[:-4] + "-right.tif", cv2.IMREAD_GRAYSCALE)


# Find size of image1
sz = img_left.shape

# Define the motion model
warp_mode = cv2.MOTION_HOMOGRAPHY

# Define a 3x3 matrices and initialize the matrix to identity
warp_matrix = np.eye(3, 3, dtype=np.float32)

# Specify the number of iterations.
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
folderpath = os.path.dirname(path)
pickle.dump(warp_matrix, open(os.path.join(folderpath, "ONI_warp_matrix.p"), "wb"))

# Test and save alignment results on beads
img_right_unit16 = cv2.imread(path[:-4] + "-right.tif", -1)
img_right_aligned = cv2.warpPerspective(
    img_right_unit16,
    warp_matrix,
    (sz[1], sz[0]),
    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
)
imwrite(path[:-4] + "-right-aligned.tif", img_right_aligned, imagej=True)
