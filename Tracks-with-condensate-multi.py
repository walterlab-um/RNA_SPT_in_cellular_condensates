print("🎯 Multi-File RNA Trajectory Analysis with Folder Selection")
from util import *
from param import *

# Folder selection
print("Please choose a folder containing 'RNA' and 'condensate' subfolders:")
root = tk.Tk()
root.withdraw()  # Hide the main window

### Changed these two lines to hardcode paths for easier running
folder_path = "/Users/samm/Documents/Coding/github/RNA_spt_in_cell_dev/data"
result_path = "/Users/samm/Documents/Coding/github/RNA_spt_in_cell_dev/result" # I added result_path to save outputs in a different folder
img_path = "/Users/samm/Documents/Coding/github/RNA_spt_in_cell_dev/result/img" # I added img_path to save images to a different folder

print(f"Selected folder: {folder_path}")

if not folder_path:
    print("❌ No folder selected. Exiting...")
    exit()

os.chdir(folder_path)

# Execute file matching and loading
print("\n🔄 Finding and matching files...")
file_pairs = find_matching_files(folder_path)


if not file_pairs:
    print("❌ No matching file pairs found!")
else:
    print(f"\n📂 Loading {len(file_pairs)} experiment datasets...")
    
    all_tracks = []
    all_condensates = []
    
    for rna_path, condensate_path, exp_name in track(file_pairs, description="Loading datasets"):
        df_tracks, df_condensates = load_dataset_pair(rna_path, condensate_path, exp_name)
        
        if df_tracks is not None and df_condensates is not None:
            all_tracks.append(df_tracks)
            all_condensates.append(df_condensates)
    
    if all_tracks and all_condensates:
        # Combine all datasets
        df_tracks_combined = pd.concat(all_tracks, ignore_index=True)
        df_condensates_combined = pd.concat(all_condensates, ignore_index=True)
        
        print(f"\n✅ Successfully loaded and combined:")
        print(f"   - {len(df_tracks_combined)} total track points")
        print(f"   - {len(df_condensates_combined)} total condensate entries")
        print(f"   - {df_tracks_combined['experiment'].nunique()} experiments")
        
        # Display summary statistics
        print("\n📊 Dataset Summary:")
        for exp in df_tracks_combined['experiment'].unique():
            exp_tracks = df_tracks_combined[df_tracks_combined['experiment'] == exp]
            exp_condensates = df_condensates_combined[df_condensates_combined['experiment'] == exp]
            print(f"   {exp}: {exp_tracks['trackID'].nunique()} tracks, "
                  f"{len(exp_condensates)} condensate entries")
    
    else:
        print("❌ No valid datasets were loaded successfully")

print("\n🎉 Data loading complete!")

from util_montage import *

# Deactivated for now to speed up testing

# # Execute the analysis
# if 'df_tracks_combined' in globals() and df_tracks_combined is not None:
#     experiment_results = analyze_interactions_all_experiments(
#         df_tracks_combined, 
#         df_condensates_combined, 
#         proximity_threshold
#     )
    
#     if experiment_results:
#         create_individual_experiment_reconstructions(
#             df_tracks_combined, 
#             df_condensates_combined, 
#             experiment_results,
#             folder_path=folder_path
#         )
#     else:
#         print("❌ No valid experiment results to plot")
# else:
#     print("❌ Combined datasets not loaded. Run the data loading block first.")

# # Execute individual montage creation with error handling
# if 'experiment_results' in globals() and experiment_results:
#     try:
#         create_individual_experiment_montages(
#             df_tracks_combined,
#             df_condensates_combined,
#             experiment_results,
#             zoom_margin=25,
#             montage_cols=4  # Adjust as needed
#         )
#     except Exception as e:
#         print(f"❌ Error in montage creation: {e}")
#         print("Try reducing montage_cols or zoom_margin if the error persists")
# else:
#     print("❌ Experiment results not available. Run the reconstruction analysis first.")

# print("\n🎉 Individual experiment analysis complete!")
# print(f"📁 All outputs saved in: {folder_path}")



##################################################
# Minimum distance calculation from RNA to the boundary of condensates 
##################################################

print("🔄 Calculating distances from RNA to condensate boundaries...")

from shapely.geometry import Point, Polygon
from shapely.strtree import STRtree

### Optimized distance calculation using spatial index
# Prepare output records
records = []

if df_tracks_combined.empty or df_condensates_combined.empty:
    print("❌ Combined datasets are empty. Skipping distance calculation.")

else:
    # Process each experiment separately
    for exp in df_tracks_combined['experiment'].unique():
        print(f"  • Processing experiment: {exp}")
        
        # Filter condensates for the current experiment once
        condensates_exp = df_condensates_combined[df_condensates_combined['experiment'] == exp]
        condensates_by_frame = condensates_exp.groupby('frame')['contour_coord'].apply(list).to_dict()

        # Get all track data for the current experiment
        tracks_exp = df_tracks_combined[df_tracks_combined['experiment'] == exp]
        
        # Iterate through each frame that has condensates
        for frame, contour_list in tqdm(condensates_by_frame.items(), desc=f"    ↳ {exp}"):
            
        # 1. PRE-PROCESS AND CLEAN POLYGONS for the current frame
            shapely_polygons = []
            for cnt_str in contour_list:
                bx, by = parse_contour_string(cnt_str)
                if len(bx) >= 3:  # A valid polygon needs at least 3 vertices
                    # Create the polygon
                    poly = Polygon(zip(bx, by))
                    
                    # Clean the polygon of any self-intersections or other invalid topology.
                    # If the polygon is already valid, this has no negative effect.
                    if not poly.is_valid:
                        poly = poly.buffer(0)

                    # Only add non-empty polygons to the list
                    if not poly.is_empty:
                        shapely_polygons.append(poly)

            # If no valid polygons could be created for this frame, skip
            if not shapely_polygons:
                continue
            
            # 2. BUILD SPATIAL INDEX
            # This index is incredibly fast to build and query
            spatial_index = STRtree(shapely_polygons)
            
            # Get all points (tracks) for the current frame
            try:
                # Note: Your original 't' seems to be frame number
                tracks_in_frame = tracks_exp[tracks_exp['t'] == frame]
            except KeyError:
                continue

            # # 3. QUERY & CALCULATE DISTANCE EFFICIENTLY
            # for idx, row in tracks_in_frame.iterrows():
            #     point = Point(row['x'], row['y'])
                            
            #     # 1. The query returns the *index* of the nearest polygon in the list.
            #     nearest_poly_index = spatial_index.nearest(point)
                
            #     # 2. Use the index to get the actual polygon object from your list.
            #     # The 'if' check handles cases where the index might be invalid.
            #     if nearest_poly_index is not None:
            #         nearest_poly = shapely_polygons[nearest_poly_index]
            #     else:
            #         continue # Skip if no polygon was found

            #     # 3. Now, 'nearest_poly' is a proper Polygon object, and this will work.
            #     min_dist = point.distance(nearest_poly)
                
            #     # Convert to desired units
            #     dist_um = min_dist * um_per_pixel
                
            #     records.append({
            #         'experiment': exp,
            #         'trackID': row['trackID'],
            #         't': frame * s_per_frame,
            #         'distance_um': dist_um
            #     })

            # 3. QUERY & CALCULATE SIGNED DISTANCE EFFICIENTLY
            for idx, row in tracks_in_frame.iterrows():
                point = Point(row['x'], row['y'])
                            
                # 1. The query returns the *index* of the nearest polygon in the list.
                nearest_poly_index = spatial_index.nearest(point)
                
                # 2. Use the index to get the actual polygon object from your list.
                if nearest_poly_index is not None:
                    nearest_poly = shapely_polygons[nearest_poly_index]
                else:
                    continue # Skip if no polygon was found

                # 3. Calculate the signed distance.
                # First, get the raw distance to the polygon's boundary (always positive).
                min_dist = point.distance(nearest_poly.boundary)
                
                # Next, check if the point is inside the polygon.
                if nearest_poly.contains(point):
                    # If it's inside, make the distance negative.
                    min_dist = -min_dist
                
                # Convert to desired units
                dist_um = min_dist * um_per_pixel
                
                records.append({
                    'experiment': exp,
                    'trackID': row['trackID'],
                    't': frame * s_per_frame,
                    'distance_um': dist_um
                })

# Create DataFrame and save
df_distances = pd.DataFrame(records)
output_csv = os.path.join(result_path, "rna_condensate_distances.csv")
df_distances.to_csv(output_csv, index=False)
print(f"✅ Saved distances to: {output_csv}")


print("🔄 Generating distance vs. time plots...")

# Load distances if not in memory
try:
    df_distances = df_distances
except NameError:
    df_distances = pd.read_csv(os.path.join(result_path, "rna_condensate_distances.csv"))

# Identify first dwell per track per experiment
first_dwell = df_distances[df_distances['distance_um'] == 0] \
    .groupby(['experiment', 'trackID'])['t'] \
    .min() \
    .reset_index() \
    .rename(columns={'t': 't_dwell'})

print(f"  • Found {len(first_dwell)} first-dwell events across all experiments")

# Merge dwell times back into distances
df_plot = df_distances.merge(first_dwell, on=['experiment', 'trackID'], how='left')

# Keep only points at and after first dwell
df_plot = df_plot[df_plot['t'] >= df_plot['t_dwell']]

print("🔄 Generating distance vs. time plots for each experiment...")

# Load distances if not in memory
try:
    df_distances = df_distances
except NameError:
    df_distances = pd.read_csv(os.path.join(result_path, "rna_condensate_distances.csv"))

# Identify first dwell per track per experiment
first_dwell = df_distances[df_distances['distance_um'] <= 0] \
    .groupby(['experiment', 'trackID'])['t'] \
    .min() \
    .reset_index() \
    .rename(columns={'t': 't_dwell'})

print(f"  • Found {len(first_dwell)} first-dwell events across all experiments")

# Merge dwell times back into distances
df_plot = df_distances.merge(first_dwell, on=['experiment', 'trackID'], how='left')
# Keep only points at and after first dwell
df_plot = df_plot[df_plot['t'] >= df_plot['t_dwell']]

# Create separate plot for each experiment
experiments = df_plot['experiment'].unique()
print(f"  • Creating {len(experiments)} separate plots...")

from scipy.signal import find_peaks # For peak detection

df_filtered = pd.DataFrame()  # To collect tracks with peak detection errors
df_error = pd.DataFrame()  # To collect tracks with peak detection errors

error_count = 0
for exp in experiments:
    print(f"    → Processing experiment: {exp}")
    
    # Filter data for this experiment
    exp_data = df_plot[df_plot['experiment'] == exp]
    
    if exp_data.empty:
        print(f"      ⚠️ No dwell data found for {exp}, skipping...")
        continue
    
    # Create figure for this experiment
    fig, ax = plt.subplots(figsize=(10, 7))
    
    track_count = 0
    # Plot each track in this experiment
    for track in np.sort(exp_data['trackID'].unique()):
        # if track_count >= 10:
        #     print(f"      ⚠️ More than 10 tracks in {exp}, limiting to first 10 for clarity.")
        #     break
        print(f"      • Plotting track {int(track)}...", end="\r")
        track_data = exp_data[exp_data['trackID'] == track]
        
        # Use find_peaks to highlight peaks with huge prominence
        x = np.array(track_data['t'] - track_data['t_dwell'])
        y = np.array(track_data['distance_um'])

        # Use find_peaks to highlight peaks with huge prominence
        try:
            peaks, _ = find_peaks(y,
                                  height=0.4, # only consider peaks above 0.4 um (arbitrary)
                                  prominence=0.7) # only consider peaks with at least 0.7 um prominence (arbitrary)
            # ax.plot(x[peaks], y[peaks], "x", color='red')
            # Continue to the next track if peak is found
            if len(peaks) >= 1:
                error_count += 1
                df_error = pd.concat([df_error, track_data], ignore_index=True)
                continue

        except Exception as e:
            print(f"      ❌ Error finding peaks for track {track} in {exp}: {e}")
            pass

        # Use different colors for better visibility
        ax.plot(x, y,
                alpha=0.5,
                linewidth=1.5,
                label=f"Track {int(track)}")
        
        df_filtered = pd.concat([df_filtered, track_data], ignore_index=True)
        track_count += 1
    
    ax.set_xlabel("Time since first dwell (s)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Distance to condensate boundary (μm)", fontsize=12, fontweight='bold')
    ax.set_title(f"RNA Distance from Condensate Boundary Over Time\n{exp} ({track_count} tracks)", 
                fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Add legend with reasonable limit
    max_legend_entries = 15
    if track_count <= max_legend_entries:
        ax.legend(fontsize=9, ncol=2, loc='upper right', framealpha=0.9)
    else:
        print(f"      ⚠️ Too many tracks ({track_count}) for legend in {exp}")
    
    # Add summary statistics as text box
    max_dist = exp_data['distance_um'].max()
    mean_dist = exp_data['distance_um'].mean()
    textstr = f'Max distance: {max_dist:.2f} μm\nMean distance: {mean_dist:.2f} μm'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save with experiment-specific filename
    safe_exp_name = exp.replace(" ", "_").replace("/", "_").replace("\\", "_")
    output_png = os.path.join(img_path, f"distance_vs_time_{safe_exp_name}.png")
    fig.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"      ✅ Saved: distance_vs_time_{safe_exp_name}.png")
    
    # plt.show()
    plt.close()  # Close to free memory

### Added section to create a separate plot for tracks with peak detection errors
# Create the plot for tracks with peak detection errors
# if not df_error.empty:
#     fig, ax = plt.subplots(figsize=(10, 7))
    
#     for exp in df_error['experiment'].unique():
#         exp_data = df_error[df_error['experiment'] == exp]
#         for track in exp_data['trackID'].unique():
#             track_data = exp_data[exp_data['trackID'] == track]
#             x = np.array(track_data['t'] - track_data['t_dwell'])
#             y = np.array(track_data['distance_um'])
#             ax.plot(x, y,
#                     alpha=0.5,
#                     linewidth=1.5)

#     ax.set_xlabel("Time since first dwell (s)", fontsize=12, fontweight='bold')
#     ax.set_ylabel("Distance to condensate boundary (μm)", fontsize=12, fontweight='bold')
#     ax.set_title(f"RNA Distance from Condensate Boundary Over Time\n(Tracks with Peak Detection Errors)", 
#                 fontsize=14, fontweight='bold')
#     ax.grid(True, linestyle='--', alpha=0.3)
    
#     plt.tight_layout()

#     # Use a safe experiment name for the error plot
#     error_exp_names = "_".join([str(e).replace(" ", "_").replace("/", "_").replace("\\", "_") for e in df_error['experiment'].unique()])
#     error_output_png_filename = f"peak_detection_errors_{error_exp_names}.png"
#     output_png = os.path.join(img_path, error_output_png_filename)
#     fig.savefig(output_png, dpi=300, bbox_inches='tight')
#     print(f"      ✅ Saved: {error_output_png_filename}")

#     # plt.show()
#     plt.close()  # Close to free memory

# Save the filtered data into CSV files based on their types
experiments_list = df_filtered['experiment'].unique()
unique_exp_list = []
for exp in experiments_list:
    # Extract part before '_FOV' or '-FOV'
    if '_FOV' in exp:
        exp_str = exp.split('_FOV')[0]
    elif '-FOV' in exp:
        exp_str = exp.split('-FOV')[0]
    else:
        exp_str = exp

    unique_exp_list.append(exp_str)

unique_exp_list = list(set(unique_exp_list))

for exp in unique_exp_list:
    print(f"  • Found unique experiment: {exp}")
    # Create a filtered DataFrame for the current experiment
    df_exp = df_filtered[df_filtered['experiment'].str.contains(exp)]
     
    # Save to CSV
    print(f"    → Saving data for experiment: {exp}")
    df_exp = df_filtered[df_filtered['experiment'].str.contains(exp)]
    output_csv = os.path.join(result_path, f"{exp}.csv")

    # Check if csv already exists
    if os.path.exists(output_csv):
        print(f"    ⚠️ File already exists, skipping: {output_csv}.csv")
    else:
        df_exp.to_csv(output_csv, index=False)
    
df_filtered.to_csv(os.path.join(result_path, "filtered_rna_condensate_distances.csv"), index=False)

print(f"Number of tracks with peak detection errors: {error_count} ({error_count/len(df_plot['trackID'].unique())*100:.1f}%)")
print(f"🎉 Created {len(experiments)} individual distance vs. time plots!")