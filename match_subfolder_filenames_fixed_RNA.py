import os
from rich.progress import track
from tkinter import filedialog
from os.path import basename


def check_file_exists(folder, experiment_name, prefix, postfix):
    out = False
    for file in os.listdir(folder):
        if file.startswith(prefix + experiment_name) & file.endswith(postfix):
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

    rna_files = [file for file in os.listdir(rna_folder) if file.endswith("-ch2.csv")]
    experiment_names = [file.rstrip("-ch2.csv") for file in rna_files]

    for experiment_name in track(experiment_names, description=basename(folder_path)):
        condensate_match = check_file_exists(
            condensate_folder, experiment_name, "", "-ch1.csv"
        )
        cell_body_match = check_file_exists(
            cell_body_folder, experiment_name, "", ".txt"
        )

        if not condensate_match:
            print(
                f"Warning: No matching file found in 'condensate' folder for {experiment_name}"
            )
        if not cell_body_match:
            print(
                f"Warning: No matching file found in 'cell_body_mannual' folder for {experiment_name}"
            )


if __name__ == "__main__":
    main()
