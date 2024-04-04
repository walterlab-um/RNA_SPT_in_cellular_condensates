import nd2reader
import numpy as np
from tifffile import imsave
import os
import tkinter as tk
from tkinter import filedialog
from rich.progress import track
import logging

# Configure logging
logging.basicConfig(level=logging.ERROR)


def convert_nd2_to_tiff(input_files):
    total_files = len(input_files)
    for file_index, input_file in enumerate(
        track(input_files, description="Converting ND2 files...", total=total_files)
    ):
        try:
            with nd2reader.ND2Reader(input_file) as images:
                # Get the number of channels and z-stacks
                num_channels = len(images.metadata["channels"])
                num_z_stacks = images.sizes["z"]

                # Get the directory of the input file
                input_dir = os.path.dirname(input_file)

                # Get the base name of the input file (without extension)
                base_name = os.path.splitext(os.path.basename(input_file))[0]

                # Iterate over each channel
                for ch in range(num_channels):
                    # Create an empty array to store the z-stack for the current channel
                    z_stack = np.zeros(
                        (num_z_stacks, images.sizes["y"], images.sizes["x"]),
                        dtype=np.uint16,
                    )

                    # Iterate over each available z-stack and store the image data
                    for z in range(num_z_stacks):
                        try:
                            z_stack[z] = images.get_frame_2D(c=ch, z=z)
                        except KeyError as e:
                            logging.warning(
                                f"Skipping frame (c={ch}, z={z}) in file: {input_file}"
                            )
                            continue

                    # Save the z-stack as a single TIFF file in the same folder as the input file
                    output_file = os.path.join(input_dir, f"{base_name}_ch{ch+1}.tif")
                    imsave(output_file, z_stack)

        except Exception as e:
            logging.error(f"Error processing file: {input_file}")
            logging.exception(e)
            continue


# Create a Tkinter root window
root = tk.Tk()
root.withdraw()  # Hide the main window

# Open a file dialog to select ND2 files
input_files = filedialog.askopenfilenames(filetypes=[("ND2 Files", "*.nd2")])

# Convert the selected ND2 files to TIFF
convert_nd2_to_tiff(input_files)
