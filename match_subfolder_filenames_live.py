import os
from rich.progress import track
from tkinter import filedialog
from os.path import basename


def check_file_exists(folder, middle_part, prefix, postfix):
    out = False
    for file in os.listdir(folder):
        if file.startswith(prefix + middle_part) & file.endswith(postfix):
            out = True
    return out


def main():
    folder_path = filedialog.askdirectory(title="Select a folder")

    if not folder_path:
        print("No folder selected. Exiting.")
        return

    rna_folder = os.path.join(folder_path, "RNA")
    condensate_folder = os.path.join(folder_path, "condensate")
    cell_body_folder = os.path.join(folder_path, "cell_body_mannual")

    rna_files = [
        file
        for file in os.listdir(rna_folder)
        if file.startswith("SPT_results_AIO-")
        and file.endswith("-right_reformatted.csv")
    ]
    middle_parts = [
        file[len("SPT_results_AIO-") : -len("-right_reformatted.csv")]
        for file in rna_files
    ]

    for middle_part in track(middle_parts, description=basename(folder_path)):
        condensate_match = check_file_exists(
            condensate_folder, middle_part, "condensates_AIO-", "left.csv"
        )
        cell_body_match = check_file_exists(cell_body_folder, middle_part, "", ".txt")

        if not condensate_match:
            print(
                f"Warning: No matching file found in 'condensate' folder for {middle_part}"
            )
        if not cell_body_match:
            print(
                f"Warning: No matching file found in 'cell_body_mannual' folder for {middle_part}"
            )


if __name__ == "__main__":
    main()
