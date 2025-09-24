# MULTI-FILE RNA TRAJECTORY ANALYSIS WITH FOLDER SELECTION
import os
from os.path import join
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import ast
import glob
from pathlib import Path
from rich.progress import track
from tkinter import filedialog as fd
import tkinter as tk
from tqdm import tqdm

def parse_list_string(list_str):
    """Safely parse string representation of list to actual list"""
    try:
        return ast.literal_eval(list_str)
    except:
        # Handle format like '[0, 1, 2, 3]'
        list_of_strings = list_str[1:-1].split(", ")
        return [float(s) for s in list_of_strings]

def convert_wide_to_long_format(df_wide):
    """Convert wide format tracks to long format for easier plotting"""
    long_data = []
    for _, row in df_wide.iterrows():
        track_id = int(row['trackID'])
        t_list = parse_list_string(row['list_of_t'])
        x_list = parse_list_string(row['list_of_x'])
        y_list = parse_list_string(row['list_of_y'])
        
        for t, x, y in zip(t_list, x_list, y_list):
            long_data.append({
                'trackID': track_id,
                't': t,
                'x': x,
                'y': y,
                'displacement_um': row.get('Displacement_um', np.nan),
                'n_steps': row.get('N_steps', len(t_list))
            })
    return pd.DataFrame(long_data)

def parse_contour_string(cnt_str):
    """Parse contour coordinate string from condensate data"""
    try:
        # Handle format like '[[x1, y1], [x2, y2], ...]'
        coords = ast.literal_eval(cnt_str)
        x = [coord[0] for coord in coords]
        y = [coord[1] for coord in coords]
        return np.array(x), np.array(y)
    except:
        try:
            # Handle alternative format
            coord_pairs = [
                tuple(map(float, coord.split(", ")))
                for coord in cnt_str[2:-2].split("], [")
            ]
            x = [coord[0] for coord in coord_pairs]
            y = [coord[1] for coord in coord_pairs]
            return np.array(x), np.array(y)
        except:
            return np.array([]), np.array([])

# File matching logic (similar to your reference code)
def find_matching_files(folder_path):
    """Find and match RNA and condensate files based on experiment names"""
    
    rna_folder = join(folder_path, "RNA")
    condensate_folder = join(folder_path, "condensate")
    
    if not os.path.exists(rna_folder):
        print(f"❌ RNA folder not found: {rna_folder}")
        return []
    
    if not os.path.exists(condensate_folder):
        print(f"❌ Condensate folder not found: {condensate_folder}")
        return []
    
    # Find RNA files
    rna_files = [
        file for file in os.listdir(rna_folder)
        if file.startswith("SPT_results_AIO") and file.endswith(".csv")
    ]
    
    # Find condensate files
    condensate_files = [
        file for file in os.listdir(condensate_folder)
        if file.startswith("condensates_AIO") and file.endswith(".csv")
    ]
    
    print(f"Found {len(rna_files)} RNA files and {len(condensate_files)} condensate files")
    
    # Extract experiment names from RNA files
    experiment_names = []
    for file in rna_files:
        # Remove prefix "SPT_results_AIO-" and suffix "-right_reformatted.csv"
        exp_name = file.replace("SPT_results_AIO-", "").replace("-right_reformatted.csv", "")
        experiment_names.append(exp_name)
    
    # Match files
    file_pairs = []
    for exp in experiment_names:
        rna_file = f"SPT_results_AIO-{exp}-right_reformatted.csv"
        rna_path = join(rna_folder, rna_file)
        
        if not os.path.exists(rna_path):
            print(f"[WARNING] RNA file not found: {rna_file}")
            continue
        
        # Find matching condensate files for this experiment
        matching_condensate_files = [
            f for f in condensate_files 
            if f.startswith(f"condensates_AIO-{exp}")
        ]
        
        if not matching_condensate_files:
            print(f"[WARNING] No condensate files found for experiment: {exp}")
            continue
        
        # Use the first matching condensate file (or you can process all)
        condensate_file = matching_condensate_files[0]
        condensate_path = join(condensate_folder, condensate_file)
        
        file_pairs.append((rna_path, condensate_path, exp))
        print(f"✅ Matched: {exp}")
    
    return file_pairs

def load_dataset_pair(rna_path, condensate_path, experiment_name):
    """Load and process a single RNA-condensate dataset pair"""
    try:
        print(f"  Loading {experiment_name}...")
        
        # Load RNA tracks
        df_tracks_wide = pd.read_csv(rna_path)
        df_tracks = convert_wide_to_long_format(df_tracks_wide)
        
        # Load condensates
        df_condensates = pd.read_csv(condensate_path)
        
        # Add experiment identifier
        df_tracks['experiment'] = experiment_name
        df_condensates['experiment'] = experiment_name
        
        print(f"    ✅ {len(df_tracks_wide)} tracks, {len(df_condensates)} condensate entries")
        return df_tracks, df_condensates
        
    except Exception as e:
        print(f"    ❌ Error loading {experiment_name}: {e}")
        return None, None