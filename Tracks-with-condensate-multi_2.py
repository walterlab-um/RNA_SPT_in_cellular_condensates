from util.util import *
from util.param import *
from util.util import *
from util.util_montage import *

debug = False

# Hardcoded path for easier running
folder_path = "/Users/samm/Documents/Coding/github/RNA_SPT_in_cellular_condensates/data" # The data is stored here (.../condensate and .../RNA w/ ... being dataset subfolders)
img_path = "/Users/samm/Documents/Coding/github/RNA_SPT_in_cellular_condensates/result/img" # Images are saved here
result_path = "/Users/samm/Documents/Coding/github/RNA_SPT_in_cellular_condensates/result" # Calculation results are stored here

os.makedirs(img_path, exist_ok=True)

# Look up all subfolders in the data folder
subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir() and not f.name.startswith('.')]
subfolders.sort()  # Sort the subfolders alphabetically

print(f"Found {len(subfolders)} subfolders in the data folder.")
i = 0
for subfolder in subfolders:
    print(f"    {i}: {subfolder.split('/')[-1]}")
    i += 1
    
    
# Change this index to select different subfolders
index = int(input("\nSelect index: "))
legend_bool = True
overwrite_trace_output = True

data_path = subfolders[index]
data_name = data_path.split("/")[-1]
print(f"\nSelected data for {data_name} in folder: {data_path}")


file_pairs = find_matching_files(data_path)

if not file_pairs:
    if debug:
        print("❌ No matching file pairs found!")
    pass
else:
    if debug:
        print(f"📂 Loading {len(file_pairs)} experiment datasets...")
    
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

        if debug:
            print(f"✅ Successfully loaded and combined:")
            print(f"   - {len(df_tracks_combined)} total track points")
            print(f"   - {len(df_condensates_combined)} total condensate entries")
            print(f"   - {df_tracks_combined['experiment'].nunique()} experiments")

        # Display summary statistics
        if debug:
            print("\n📊 Dataset Summary:")
        for exp in df_tracks_combined['experiment'].unique():
            exp_tracks = df_tracks_combined[df_tracks_combined['experiment'] == exp]
            exp_condensates = df_condensates_combined[df_condensates_combined['experiment'] == exp]
            if debug:
                print(f"   {exp}: {exp_tracks['trackID'].nunique()} tracks, "
                      f"{len(exp_condensates)} condensate entries")
    else:
        if debug:
            print("❌ No valid datasets were loaded successfully")
        pass

if debug:
    print("\n🎉 Data loading complete!")
    

print("🔄 Calculating distances from RNA to condensate boundaries...")

from shapely.geometry import Point, Polygon
from shapely.strtree import STRtree

### Optimized distance calculation using spatial index
output_csv = os.path.join(result_path, f"trace_csv/all/{data_name}_rna_condensate_distances.csv")

def calculate_rna_condensate_distances(df_tracks_combined,
                                       df_condensates_combined,
                                       output_csv,
                                       overwrite_save=False,
                                       um_per_pixel=um_per_pixel,
                                       s_per_frame=s_per_frame,
                                       condensate_detection_threshold=2,
                                       ):
    """
    Calculates the signed distance from RNA tracks to the nearest condensate boundary.
    If a track point is found inside a condensate, it "locks" to that condensate for subsequent frames.

    Args:
        df_tracks_combined (pd.DataFrame): DataFrame with combined track data.
        df_condensates_combined (pd.DataFrame): DataFrame with combined condensate data.
        output_csv (str): Path to save the output CSV file.
        um_per_pixel (float): Conversion factor from pixels to micrometers.
        s_per_frame (float): Conversion factor from frames to seconds.

    Returns:
        pd.DataFrame: DataFrame with calculated distances.
    """
    # Check if output CSV already exists
    if overwrite_save is False:
        if os.path.exists(output_csv):
            print(f"⚠️ Output CSV '{output_csv}' already exists. Skipping distance calculation to avoid overwriting.")
            return pd.read_csv(output_csv)

    if df_tracks_combined.empty or df_condensates_combined.empty:
        print("❌ Combined datasets are empty. Skipping distance calculation.")
        return pd.DataFrame()

    records = []
    # Process each experiment separately
    unique_experiments = df_tracks_combined['experiment'].unique()
    unique_experiments.sort()
    print(f"  • Found {len(unique_experiments)} unique experiments to process.")
    
    for exp_idx, exp in enumerate(unique_experiments):
        print(f"  • Processing experiment: {exp} ({exp_idx + 1}/{len(unique_experiments)})")
        
        # Filter data for the current experiment
        condensates_exp = df_condensates_combined[df_condensates_combined['experiment'] == exp]
        tracks_exp = df_tracks_combined[df_tracks_combined['experiment'] == exp]

        # Pre-process condensates for all frames in the experiment
        condensates_by_frame = {}
        for frame, group in condensates_exp.groupby('frame'):
            shapely_polygons = []
            for cnt_str in group['contour_coord']:
                bx, by = parse_contour_string(cnt_str)
                if len(bx) >= 3:
                    poly = Polygon(zip(bx, by))
                    if not poly.is_valid:
                        poly = poly.buffer(0)
                    if not poly.is_empty:
                        shapely_polygons.append(poly)
            
            if shapely_polygons:
                condensates_by_frame[frame] = {
                    'polygons': shapely_polygons,
                    'index': STRtree(shapely_polygons)
                }

        # Group tracks by trackID
        tracks_by_id = tracks_exp.groupby('trackID')

        # Iterate through each track in the experiment
        for track_id, track_df in tqdm(tracks_by_id, desc=f"    ↳ Tracks in {exp}"):
            locked_condensate = None # Initialize locked condensate as None
            
            # Sort track by time
            track_df = track_df.sort_values('t')

            for _, row in track_df.iterrows():
                frame = row['t']
                point = Point(row['x'], row['y'])
                min_dist = np.nan
                
                # If locked, calculate distance to the locked condensate
                if locked_condensate is not None:
                    # Check if the locked condensate still exists in the current frame
                    # This simple check assumes condensate identity is maintained by location.
                    # A more robust method would need condensate tracking.
                    if frame in condensates_by_frame:
                        spatial_index = condensates_by_frame[frame]['index']
                        polygons = condensates_by_frame[frame]['polygons']
                        
                        nearest_poly_idx = spatial_index.nearest(point)
                        if nearest_poly_idx is not None:
                            candidate_poly = polygons[nearest_poly_idx]
                            if candidate_poly.distance(locked_condensate) <= condensate_detection_threshold: # 2 frames tolerance
                                locked_condensate = candidate_poly # Update to current frame's polygon
                            else:
                                nearest_poly_to_locked_index = spatial_index.nearest(locked_condensate.centroid)
                                if nearest_poly_to_locked_index is not None:
                                    candidate_poly_to_locked = polygons[nearest_poly_to_locked_index]
                                    if candidate_poly_to_locked.distance(locked_condensate) <= condensate_detection_threshold:
                                        locked_condensate = candidate_poly_to_locked
                                        
                    min_dist = point.distance(locked_condensate.boundary)
                    if locked_condensate.contains(point):
                        min_dist = -min_dist

                else: # If locked condensate is None
                    # If not locked, find the nearest condensate
                    if frame in condensates_by_frame:
                        spatial_index = condensates_by_frame[frame]['index']
                        polygons = condensates_by_frame[frame]['polygons']
                        nearest_poly_idx = spatial_index.nearest(point)
                        if nearest_poly_idx is not None:
                            nearest_poly = polygons[nearest_poly_idx]
                            min_dist = point.distance(nearest_poly.boundary)
                            
                            if nearest_poly.contains(point):
                                min_dist = -min_dist
                                locked_condensate = nearest_poly # Lock onto this condensate

                if not np.isnan(min_dist):
                    dist_um = min_dist * um_per_pixel
                    contour_area = locked_condensate.area if locked_condensate is not None else np.nan
                    contour_coords = locked_condensate.centroid if locked_condensate is not None else np.nan
                    records.append({
                        'experiment': exp,
                        'trackID': track_id,
                        't': frame * s_per_frame,
                        'distance_um': dist_um,
                        'contour_area': contour_area,
                        'contour_coords': contour_coords
                    })

    # Create DataFrame and save
    df_distances = pd.DataFrame(records)
    if not df_distances.empty:
        df_distances.to_csv(output_csv, index=False)
        print(f"✅ Saved distances to: {output_csv}")
    else:
        print("No records were generated, CSV not saved.")
        
    return df_distances

# Call the function
df_distances = calculate_rna_condensate_distances(
    df_tracks_combined, 
    df_condensates_combined,
    output_csv,
    overwrite_save=overwrite_trace_output
)

print("🔄 Generating distance vs. time plots for each experiment...")

# Load distances if not in memory
try:
    df_distances = df_distances
except NameError:
    df_distances = pd.read_csv(output_csv)

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
experiments.sort()
print(f"  • Creating {len(experiments)} separate plots...")

from scipy.signal import find_peaks # For peak detection

df_filtered = pd.DataFrame()  # To collect tracks with peak detection errors
df_error = pd.DataFrame()  # To collect tracks with peak detection errors
df_outlier = pd.DataFrame()  # To collect tracks with peak detection errors

# Maximum jump per frame is 3 pixel frames 
max_jump = 3 * um_per_pixel / s_per_frame

error_count = 0
for exp in experiments:
    print(f"    → Processing experiment: {exp}")
    
    # Filter data for this experiment
    exp_data = df_plot[df_plot['experiment'] == exp]
    
    if exp_data.empty:
        print(f"      ⚠️ No dwell data found for {exp}, skipping...")
        continue
    
    # Create figure for this experiment
    fig, ax = plt.subplots(2, 1, figsize=(10, 14))
    
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
        
        dx = np.diff(x)
        dy = np.diff(y)
        # Calculate derivative (dy/dx)
        derivative = dy / dx
        
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
        ax[0].plot(x, y,
                alpha=0.5,
                linewidth=1.5,
                label=f"Track {int(track)}")

        ax[0].scatter(x, y,
                   s=10,
                   alpha=0.7,
                   zorder=10)

        # Plot the derivative if it exists
        if 'derivative' in locals():
            ax[1].plot(x[:-1], derivative, alpha=0.5, linewidth=1.5, label=f"Track {int(track)}")
            
            if any(abs(derivative) > max_jump):
                df_outlier = pd.concat([df_outlier, track_data], ignore_index=True)
        
        df_filtered = pd.concat([df_filtered, track_data], ignore_index=True)
        track_count += 1
    
    ax[0].axhline(0, color='gray', linestyle='--', linewidth=1, zorder=12)
    
    ax[1].axhline(max_jump, color='orange', linestyle='--', linewidth=1, zorder=12)
    ax[1].axhline(-max_jump, color='orange', linestyle='--', linewidth=1, zorder=12)

    ax[0].set_xlabel("")
    ax[0].set_ylabel("Distance to condensate boundary (μm)", fontsize=12, fontweight='bold')
    ax[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    
    ax[1].set_xlabel("Time since first dwell (s)", fontsize=12, fontweight='bold')
    ax[1].set_ylabel("Distance Derivative (μm/s)", fontsize=12, fontweight='bold')
    
    
    ax[0].set_title(f"RNA Distance from Condensate Boundary Over Time\n{exp} ({track_count} tracks)", 
                fontsize=14, fontweight='bold')
    ax[0].grid(True, linestyle='--', alpha=0.3)

    # # Add legend with reasonable limit
    # max_legend_entries = 15
    # if track_count <= max_legend_entries:
    #     ax[0].legend(fontsize=9, ncol=2, loc='upper right', framealpha=0.9)
    # else:
    #     print(f"      ⚠️ Too many tracks ({track_count}) for legend in {exp}")
    
    # Add summary statistics as text box
    max_dist = exp_data['distance_um'].max()
    mean_dist = exp_data['distance_um'].mean()
    textstr = f'Max distance: {max_dist:.2f} μm\nMean distance: {mean_dist:.2f} μm'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    
    ax[0].text(0.02, 0.98, textstr, transform=ax[0].transAxes, fontsize=10, verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save with experiment-specific filename
    safe_exp_name = exp.replace(" ", "_").replace("/", "_").replace("\\", "_")
    output_png = os.path.join(img_path, f"distance_vs_time_{safe_exp_name}.png")
    fig.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"      ✅ Saved: distance_vs_time_{safe_exp_name}.png")
    
    # plt.show()
    plt.close()  # Close to free memory
    
if not df_outlier.empty:
    n_outliers = df_outlier['trackID'].nunique()
    print(f"🔔 Found {n_outliers} tracks with outlier jumps exceeding {max_jump:.2f} μm. ({n_outliers/len(df_plot['trackID'].unique())*100:.1f}%)")

### Added section to create a separate plot for tracks with peak detection errors
# Create the plot for tracks with peak detection errors
if not df_outlier.empty:
    all_tracks = []
    for exp in df_outlier['experiment'].unique():
        exp_data = df_outlier[df_outlier['experiment'] == exp]
        for track in exp_data['trackID'].unique():
            all_tracks.append((exp, track))

    plot_count = 0
    for i in range(0, len(all_tracks), 30):
        plot_count += 1
        chunk_of_tracks = all_tracks[i:i + 30]
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Get unique experiment names for this chunk for the filename
        chunk_experiments = set()

        for exp, track_id in chunk_of_tracks:
            track_data = df_outlier[(df_outlier['experiment'] == exp) & (df_outlier['trackID'] == track_id)]
            x = np.array(track_data['t'] - track_data['t_dwell'])
            y = np.array(track_data['distance_um'])
            ax.plot(x, y, alpha=0.5, linewidth=1.5, label=f"Track {track_id}")
            chunk_experiments.add(exp)

        ax.set_xlabel("Time since first dwell (s)", fontsize=12, fontweight='bold')
        ax.set_ylabel("Distance to condensate boundary (μm)", fontsize=12, fontweight='bold')
        ax.set_title(f"RNA Distance from Condensate Boundary Over Time\n(Outlier Tracks, Part {plot_count})", 
                    fontsize=14, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        
        # Convert set to a sorted list to get the first and last experiment names
        sorted_chunk_experiments = sorted(list(chunk_experiments))
        chunk_experiments_for_filename = (sorted_chunk_experiments[0], sorted_chunk_experiments[-1]) if sorted_chunk_experiments else []
        
        # Create a safe filename for the plot
        error_exp_names = "_".join(sorted([str(e).replace(" ", "_").replace("/", "_").replace("\\", "_") for e in chunk_experiments_for_filename]))
        
        # Make sure the outliers directory exists
        outlier_img_path = os.path.join(img_path, "outliers")
        os.makedirs(outlier_img_path, exist_ok=True)
        
        error_output_png_filename = f"outliers_{error_exp_names}_part_{plot_count}.png"
        output_png = os.path.join(outlier_img_path, error_output_png_filename)
        
        fig.savefig(output_png, dpi=300, bbox_inches='tight')
        print(f"      ✅ Saved: {error_output_png_filename}")

        plt.close()  # Close to free memory
    
    # Plot the snapshot
    # Create snapshots for outlier tracks
    snapshot_save_path = os.path.join(img_path, "outliers", "snapshots")
    os.makedirs(snapshot_save_path, exist_ok=True)
    
    snapshot_tracks = all_tracks[:10]
    
    for exp, track_id in snapshot_tracks:
        print(f"  • Creating snapshot for outlier track: {exp}, Track ID: {track_id}")
        
        # Get the full track data from the combined dataframe
        track_df = df_tracks_combined[(df_tracks_combined['experiment'] == exp) & (df_tracks_combined['trackID'] == track_id)]
        
        # Get the condensate data for the experiment
        condensate_df = df_condensates_combined[df_condensates_combined['experiment'] == exp]
        
        # Get the outlier trace data
        trace_df = df_outlier[(df_outlier['experiment'] == exp) & (df_outlier['trackID'] == track_id)]
        
        if track_df.empty or condensate_df.empty or trace_df.empty:
            print(f"      ⚠️ Could not find all necessary data for outlier track {track_id} in {exp}. Skipping snapshot.")
            continue
            
        # Analyze interactions
        try:
            experiment_result = analyze_interactions(track_df, condensate_df, proximity_threshold=1)
        except Exception as e:
            print(f"      ❌ Error analyzing interactions for outlier track {track_id} in {exp}: {e}")
            continue

        # Create the snapshot
        try:
            safe_exp_name = exp.replace(" ", "_").replace("/", "_").replace("\\", "_")
            snapshot_filename = f"snapshot_{safe_exp_name}_track_{track_id}.png"
            save_path = os.path.join(snapshot_save_path, snapshot_filename)
            
            # plot_trajectory_snapshots(df_tracks=track_df,
            #                           df_condensates=condensate_df,
            #                           exp_results=experiment_result,
            #                           trace_df=trace_df,
            #                           window_size_um=1.0,
            #                           um_per_pixel=um_per_pixel,
            #                           gradient_color=True,
            #                           show_interaction=True,
            #                           show_condensate_movements=True,
            #                           distance_by_interaction=True,
            #                           condensate_detection_threshold=2,
            #                           save_path=save_path
            #                           )
            
            # create_trajectory_movies(df_tracks=track_df,
            #                          df_condensates=condensate_df,
            #                          exp_results=experiment_result,
            #                          trace_df=trace_df,
            #                          window_size_um=3.0,
            #                          um_per_pixel=um_per_pixel,
            #                          gradient_color=True,
            #                          show_interaction=True,
            #                          show_condensate_movements=True,
            #                          distance_by_interaction=True,
            #                          save_path=save_path,
            #                          last_n_frames=5,
            #                          )
            print(f"      ✅ Saved snapshot: {snapshot_filename}")
        except Exception as e:
            print(f"      ❌ Error creating snapshot for outlier track {track_id} in {exp}: {e}")
            
    print("🎉 Finished creating outlier snapshots.")
    

if not df_error.empty:
    all_tracks = []
    for exp in df_error['experiment'].unique():
        exp_data = df_error[df_error['experiment'] == exp]
        for track in exp_data['trackID'].unique():
            all_tracks.append((exp, track))

    plot_count = 0
    for i in range(0, len(all_tracks), 20):
        plot_count += 1
        chunk_of_tracks = all_tracks[i:i + 20]

        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Get unique experiment names for this chunk for the filename
        chunk_experiments = set()

        for exp, track_id in chunk_of_tracks:
            track_data = df_error[(df_error['experiment'] == exp) & (df_error['trackID'] == track_id)]
            x = np.array(track_data['t'] - track_data['t_dwell'])
            y = np.array(track_data['distance_um'])
            ax.plot(x, y, alpha=0.5, linewidth=1.5, label=f"Track {track_id}")
            chunk_experiments.add(exp)

        ax.set_xlabel("Time since first dwell (s)", fontsize=12, fontweight='bold')
        ax.set_ylabel("Distance to condensate boundary (μm)", fontsize=12, fontweight='bold')
        ax.set_title(f"RNA Distance from Condensate Boundary Over Time\n(Tracks with Peak Detection Errors, Part {plot_count})", 
                    fontsize=14, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3)
        
        plt.tight_layout()

        # Convert set to a sorted list to get the first and last experiment names
        sorted_chunk_experiments = sorted(list(chunk_experiments))
        chunk_experiments_for_filename = (sorted_chunk_experiments[0], sorted_chunk_experiments[-1]) if sorted_chunk_experiments else []
        
        # Create a safe filename for the plot
        error_exp_names = "_".join(sorted([str(e).replace(" ", "_").replace("/", "_").replace("\\", "_") for e in chunk_experiments_for_filename]))
        
        # Make sure the errors directory exists
        error_img_path = os.path.join(img_path, "errors")
        os.makedirs(error_img_path, exist_ok=True)
        
        error_output_png_filename = f"peak_detection_errors_{error_exp_names}_part_{plot_count}.png"
        output_png = os.path.join(error_img_path, error_output_png_filename)
        
        fig.savefig(output_png, dpi=300, bbox_inches='tight')
        print(f"      ✅ Saved: {error_output_png_filename}")

        plt.close()  # Close to free memory


# Save the filtered data into CSV files based on their types
experiments_list = df_filtered['experiment'].unique()
unique_exp_list = []

for exp in experiments_list:
    # Extract part before '_FOV' or '-FOV'
    if '_FOV' in exp:
        exp_str = str(exp.split('_FOV')[0])
    elif '-FOV' in exp:
        exp_str = str(exp.split('-FOV')[0])
    else:
        exp_str = str(exp)
    
    # Takes in the name of the experiment without the date
    exp_str = exp_str.split('-')[-1]
    
    if '_replicate' in exp_str:
        exp_str = exp_str.split('_replicate')[0]
    
    unique_exp_list.append(exp_str)

unique_exp_list = list(set(unique_exp_list))


save_overwrite = True
distances_output_csv = os.path.join(result_path, f"{data_name}_filtered_rna_condensate_distances.csv")

if os.path.exists(distances_output_csv):
    if save_overwrite is False:
        print(f"    ⚠️ File already exists, skipping: {distances_output_csv}")
    else:
        df_filtered.to_csv(distances_output_csv, index=False)
        print(f"    ✅ Overwrote existing file: {distances_output_csv}")
else:
    df_filtered.to_csv(distances_output_csv, index=False)


override_save = True

for exp in unique_exp_list:
    print(f"  • Found unique experiment: {exp}")
    # Create a filtered DataFrame for the current experiment
    exp_result_df = df_filtered[df_filtered['experiment'].str.contains(exp)]
     
    # Save to CSV
    print(f"    → Saving data for experiment: {exp}")
    t_dwell_csv = os.path.join(result_path,f"trace_csv/t_dwell/t-dwell_{exp}.csv")
    output_csv = os.path.join(result_path, f"trace_csv/raw/{exp}.csv")
    
    # For the dwell info, keep only the dwell time, experiment and trackID
    t_dwell_df = exp_result_df.drop(['t', 'distance_um'], axis=1)
    t_dwell_df = t_dwell_df.drop_duplicates()
    
    # For the distance traces, keep only t and distance_um
    trace_df = exp_result_df.drop(['t_dwell'], axis=1)
    
    # Check if csv already exists
    if os.path.exists(output_csv) and not override_save:
        print(f"    ⚠️ File already exists, skipping: {output_csv}")
    elif override_save:
        print(f"    ✅ Saving distance traces to: {output_csv}")
        print(f"    ✅ Saving dwell info to: {t_dwell_csv}")
        t_dwell_df.to_csv(t_dwell_csv, index=False)
        trace_df.to_csv(output_csv, index=False)

print(f"Number of tracks with peak detection errors: {error_count} ({error_count/len(df_plot['trackID'].unique())*100:.1f}%)")