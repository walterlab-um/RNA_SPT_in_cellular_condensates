# %--- coding: utf-8 ---

import os
import glob
import numpy as np
import pandas as pd
import math

import matplotlib.pyplot as plt
import seaborn as sns


# Clustering analysis utilities
import umap
import hdbscan
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


common_time = np.linspace(0, 20, 100)  # Common time vector for interpolation


def create_synthetic_data():
    """Generates a few types of trajectories for demonstration."""
    if not os.path.exists('sample_trajectories'):
        os.makedirs('sample_trajectories')
    
    # Type 1: Stays low
    for i in range(20):
        time = np.linspace(0, 20, np.random.randint(50, 100))
        distance = 0.1 + np.random.rand(len(time)) * 0.1
        pd.DataFrame({'time': time, 'distance': distance}).to_csv(f'sample_trajectories/low_dwell_{i}.csv', index=False)

    # Type 2: Spikes and decays
    for i in range(20):
        time = np.linspace(0, 20, np.random.randint(50, 100))
        distance = 1.2 * np.exp(-time / 5) + np.random.rand(len(time)) * 0.15
        pd.DataFrame({'time': time, 'distance': distance}).to_csv(f'sample_trajectories/spike_decay_{i}.csv', index=False)
        
    # Type 3: Noisy and high
    for i in range(20):
        time = np.linspace(0, 20, np.random.randint(50, 100))
        distance = 0.6 + np.random.rand(len(time)) * 0.4
        pd.DataFrame({'time': time, 'distance': distance}).to_csv(f'sample_trajectories/noisy_high_{i}.csv', index=False)


################################################
# File Loading and Preprocessing
################################################

# --- Function to deconvolute trajectory data ---
def deconvolute_trajectory_data(file_path):
    """Deconvolutes a trajectory CSV file into individual tracks.

    Args:
        file_path (str): Path to the input CSV file.

    Returns:
        list: A list of DataFrames, each containing a single trajectory.
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    df_list = []
    
    # The csv contains multiple trajectories identified by 'trackID'
    experiments = df['experiment'].unique()
    for exp in experiments:
        exp_data = df[df['experiment'] == exp]
        for track in np.sort(exp_data['trackID'].unique()):
            track_data = exp_data[exp_data['trackID'] == track]

            # Sort the time points
            track_data = track_data.sort_values(by='t')
            track_data.loc[:, 't'] = track_data['t'] - track_data['t'].min()  # Normalize time to start at 0
            
            # Save to new CSV file
            df_list.append(track_data)

    return df_list

# --- Function to Load and Preprocess Data ---
def load_and_prepare_trajectories(folder_path, file_pattern='2x', common_time=common_time):
    """
    Loads all trajectory CSVs from a folder and interpolates them to a fixed length.
    
    Args:
        folder_path (str): Path to the folder with trajectory files.
        n_points (int): The number of points to standardize each trajectory to.
        file_pattern (str): A string to filter the csv files by their pattern.

    Returns:
        tuple: A tuple containing the processed data matrix (numpy array) and a list of original trajectories.
    """
    file_paths = glob.glob(os.path.join(folder_path, f'*{file_pattern}*.csv'))
    print(f"Found {len(file_paths)} files matching pattern '{file_pattern}' in {folder_path}.")
    processed_traces = []
    original_traces = []
    
    for path in file_paths:
        df = deconvolute_trajectory_data(path)
        # Interpolate the distance values onto the common time axis
        original_traces.append(df)
        print(f"Loaded {os.path.basename(path)} with {len(df)} tracks.")
        
        for temp_df in df:
            interp_distance = np.interp(common_time, temp_df['t'], temp_df['distance_um'])
            processed_traces.append(interp_distance)
    
    if len(file_paths) == 1:
        original_traces = original_traces[0]  # Unwrap if only one file
    
    return np.array(processed_traces), original_traces


# --- Function to Preprocess The Data ---

def find_duration_histogram(traces,
                            show_plot=False):
    """Finds and plots the duration histogram of the given traces.

    Args:
        traces (list): List of DataFrames containing trajectory data.
        show_plot (bool, optional): Whether to display the plot. Defaults to True.

    Returns:
        list: A list containing the first time points, last time points, and durations.
    """
    # Initialize lists to store first time and last time
    first_time_list, last_time_list = [], []

    # Loop through each trajectory DataFrame
    for trace_df in traces:
        first_time = trace_df['t'].iloc[0]
        last_time = trace_df['t'].iloc[-1]

        first_time_list.append(first_time)
        last_time_list.append(last_time)
    
    if show_plot:
        fig, ax = plt.subplots(1, 2, figsize=(14, 7))

        sns.histplot(first_time_list, bins=20, kde=True, ax=ax[0], stat='density')
        ax[0].set_title('First Time Point Distribution')
        ax[0].set_xlabel('Time (s)')
        ax[0].set_ylabel('Density')

        sns.histplot(last_time_list, bins=20, kde=True, ax=ax[1], stat='density')
        ax[1].set_title('Last Time Point Distribution')
        ax[1].set_xlabel('Time (s)')
        ax[1].set_ylabel('Density')

        plt.tight_layout()
        plt.show()
    
    return first_time_list, last_time_list



# --- Function to Plot CDF and Find Elbow Point ---
def find_trajectory_elbow_point(traces,
                                show_plot=True):
    """Finds the elbow point in the CDF of last time points to determine t_max.
    
    Args:
        traces (list): List of DataFrames containing trajectory data.
        show_plot (bool, optional): Whether to display the plot. Defaults to True.
    Returns:
        int: The determined t_max value.
    """
    _, last_time_list = find_duration_histogram(traces, show_plot=False)
    
    # Find the elbow point in the CDF
    sorted_last_times = np.sort(last_time_list)
    n = len(sorted_last_times)
    x = np.arange(n)
    y = sorted_last_times
    # Calculate the line from first to last point
    line_start = np.array([0, y[0]])
    line_end = np.array([n-1, y[-1]])
    line_vec = line_end - line_start
    line_vec = line_vec / np.linalg.norm(line_vec)

    # Calculate distances from each point to the line
    distances = []
    for i in range(n):
        point = np.array([i, y[i]])
        point_vec = point - line_start
        proj_length = np.dot(point_vec, line_vec)
        proj_point = line_start + proj_length * line_vec
        distance = np.linalg.norm(point - proj_point)
        distances.append(distance)

    elbow_index = np.argmax(distances)
    elbow_value = sorted_last_times[elbow_index]
    
    if show_plot:
        fig, ax = plt.subplots(figsize=(10, 7))

        # Generate the CDF for the last time points and duration
        sns.ecdfplot(last_time_list, ax=ax)
        ax.axvline(elbow_value, color='red', linestyle='--', label=f'Elbow at {elbow_value:.2f}s')
        ax.legend()

        ax.set_title('CDF of Last Time Points')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Cumulative Density')
        ax.set_xlim(0, 20)

        plt.tight_layout()
        plt.show()
    else:
        pass
    print(f"Elbow point at index {elbow_index} with last time {elbow_value:.2f}s")
    
    t_max = int(elbow_value)
    print(f"Setting t_max to {t_max} seconds based on CDF analysis.")

    return t_max


def filter_and_interpolate_traces(traces,
                                  t_max=None,
                                  show_plot=True):
    """Filters and interpolates traces to a common time axis up to t_max.

    Args:
        traces (list): List of DataFrames containing trajectory data.
        common_time (numpy array): The common time axis for interpolation.
        t_max (int, optional): The maximum time to consider for filtering. Defaults to None.
        show_plot (bool, optional): Whether to display the plot. Defaults to True.

    Returns:
        tuple: A tuple containing the filtered data matrix, filtered traces,
               length matrix, and interpolated length matrix.
    """
    
    
    if t_max is None:
        t_max = find_trajectory_elbow_point(traces, show_plot=show_plot)

    common_time = np.linspace(0, t_max, int(t_max*10)) # 0.1 s intervals to match the camera's frame rate

    length_matrix = []
    interp_length_matrix = []
    filtered_data_matrix = []
    filtered_traces = []

    for trace_df in traces:
        if trace_df['t'].max() < t_max:
            continue  # Skip this trace entirely
        else:
            trace_df = trace_df[trace_df['t'] <= t_max]  # Truncate to t_max
            filtered_traces.append(trace_df)
            
            # trace_df = trace_df.drop(columns=['experiment'], errors='ignore')  # Drop "experiment" column if exists
            distance = np.array(trace_df['distance_um'].values)
            interp_distance = np.interp(common_time, trace_df['t'], distance)
            
            filtered_data_matrix.append(interp_distance)
            length_matrix.append(len(distance))
            interp_length_matrix.append(len(interp_distance))

    data_matrix = np.array(filtered_data_matrix)
    length_matrix = np.array(length_matrix)
    interp_length_matrix = np.array(interp_length_matrix)
    print(f"Filtered data matrix shape: {np.array(data_matrix).shape}")
    print(f"Filtered traces shape: {len(filtered_traces)}")
    # Check the length of the trace after filtering and interpolation
    if show_plot:
        fig, ax = plt.subplots(1, 2, figsize=(14, 7))

        bin_width = 1
        filter_bins = (np.arange(0, length_matrix.max() + bin_width, bin_width) - 0.5)

        sns.histplot(length_matrix, bins=filter_bins, kde=False, ax=ax[0], stat='probability')

        ax[0].set_title('Length of Traces After Filtering')
        ax[0].set_xlabel('Number of Time Points')
        ax[0].set_ylabel('Fraction')

        interp_bins = 10

        sns.histplot(interp_length_matrix, bins=interp_bins, kde=False, ax=ax[1], stat='probability')
        ax[1].set_title('Length of Interpolated Traces')
        ax[1].set_xlabel('Number of Time Points')
        ax[1].set_ylabel('Fraction')

        plt.show()
        
    return data_matrix, filtered_traces, length_matrix, interp_length_matrix



################################################
# Optimizations for KMeans Clustering
################################################

def optimal_kmeans(data, max_k=10):
    """
    Determines the optimal number of clusters for KMeans using multiple methods.
    
    Args:
        data (numpy array): The input data for clustering.
        max_k (int): The maximum number of clusters to test.
    Returns:
        dict: Dictionary containing optimal k from different methods.
    """
    sse = []
    silhouette_scores = []
    K = range(2, max_k + 1)
    
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)
        silhouette_avg = silhouette_score(data, kmeans.labels_)
        silhouette_scores.append(silhouette_avg)
    
    # Plot the elbow method
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('SSE', color=color)
    ax1.plot(K, sse, 'o-', color=color, label='SSE')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Silhouette Score', color=color)
    ax2.plot(K, silhouette_scores, 's--', color=color, label='Silhouette Score')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Different methods to find optimal k
    # Method 1: Highest silhouette score
    optimal_k_silhouette = K[np.argmax(silhouette_scores)]
    
    # Method 2: Elbow method (using knee detection)
    # Simple elbow detection: find point with maximum curvature
    sse_diff = np.diff(sse)
    sse_diff2 = np.diff(sse_diff)
    if len(sse_diff2) > 0:
        optimal_k_elbow = K[np.argmax(sse_diff2) + 2]  # +2 because of double diff
    else:
        optimal_k_elbow = K[0]
    
    # Method 3: Silhouette score threshold (>0.5 is good, >0.7 is excellent)
    good_silhouette_indices = np.where(np.array(silhouette_scores) > 0.5)[0]
    if len(good_silhouette_indices) > 0:
        optimal_k_threshold = K[good_silhouette_indices[0]]  # First k with good silhouette
    else:
        optimal_k_threshold = optimal_k_silhouette
    
    # Add vertical lines for different optimal k values
    ax1.axvline(x=optimal_k_silhouette, color='red', linestyle='--', alpha=0.7, 
                label=f'Max Silhouette k={optimal_k_silhouette}')
    ax1.axvline(x=optimal_k_elbow, color='green', linestyle='--', alpha=0.7, 
                label=f'Elbow k={optimal_k_elbow}')
    ax1.axvline(x=optimal_k_threshold, color='purple', linestyle='--', alpha=0.7, 
                label=f'Threshold k={optimal_k_threshold}')
    
    plt.title('Elbow Method and Silhouette Scores for KMeans')
    
    # Create combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.show()
    
    # Return multiple recommendations
    results = {
        'max_silhouette': optimal_k_silhouette,
        'elbow': optimal_k_elbow, 
        'threshold': optimal_k_threshold,
        'silhouette_scores': silhouette_scores,
        'sse': sse
    }
    
    return results




################################################
# UMAP and KMeans Clustering
################################################
def umap_kmeans_clustering(data: pd.DataFrame,
                           n_neighbors: int = 15,
                           min_dist: float = 0.1,
                           n_clusters: int = 5):
    """
    Performs UMAP dimensionality reduction followed by KMeans clustering.
    
    Args:
        data (numpy array): The input data for clustering.
        n_clusters (int): The number of clusters for KMeans.
        n_neighbors (int): The number of neighbors for UMAP.
        min_dist (float): The minimum distance parameter for UMAP.
        
    Returns:
        tuple: UMAP reducer, KMeans model, UMAP embedding, KMeans labels
    """
    # UMAP reduction
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    embedding = reducer.fit_transform(data)
    
    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(embedding)
    
    return reducer, kmeans, embedding, kmeans.labels_












def plot_trajectories(cluster_labels,
                      filtered_traces,
                      color_map,
                      t_max,
                      n_neighbors
                      ):
    """Plot trajectories for each cluster.

    Args:
        cluster_labels (_type_): _description_
        filtered_traces (_type_): _description_
        color_map (_type_): _description_
        t_max (_type_): _description_
        n_neighbors (_type_): _description_

    Returns:
        _type_: _description_
    """
    closest_trace = {}

    # Get the unique cluster labels that we need to plot
    unique_labels = np.unique(cluster_labels)
    n_clusters = len(unique_labels)
    n_cols = 2 # Number of plots you want in each row
    n_rows = math.ceil(n_clusters / n_cols) # Calculate rows needed
    # Create a figure that is large enough to hold the grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows), sharey=True)

    # Flatten the 2D array of axes into a 1D array for easy looping
    # This also handles the case where n_rows is 1
    axes = axes.flatten()

    # --- 3. Loop Through Clusters and Plot Trajectories ---
    cluster_above_threshold = {label: 0 for label in unique_labels}
    for i, label in enumerate(unique_labels):
        ax = axes[i] # Select the subplot for the current cluster
        
        # Find all trajectories that belong to this cluster
        indices = np.where(cluster_labels == label)[0]
        
        distance_list = []
        count_above_threshold = 0
        # Plot each trajectory in this cluster
        for trace_idx in indices:
            trace_df = filtered_traces[trace_idx]
            
            if trace_df['distance_um'].max() > 0.5:
                count_above_threshold += 1
                ax.plot(trace_df['t'], trace_df['distance_um'], color=color_map[label], alpha=0.3, zorder=11)
                
            else:
                ax.plot(trace_df['t'], trace_df['distance_um'], color='grey', alpha=0.1, zorder=1)
                distance_list.append(np.interp(common_time, trace_df['t'], trace_df['distance_um']))
        
        # Plot the average trajectory for this cluster
        if distance_list:
            distance_list = np.array(distance_list)
            # Calculate the average trajectory
            mean_distance = np.mean(distance_list, axis=0)
            # Plot the mean trajectory with a thicker line
            ax.plot(common_time, mean_distance, color='black', linewidth=2, label='Mean Trajectory', zorder=20, linestyle='--')
            
            # Find the closest trajectory to the mean
            mean_idx = np.argmin(np.linalg.norm(distance_list - mean_distance, axis=1))
            
            closest_trace[label] = filtered_traces[indices[mean_idx]] # Store the closest trace (as a DataFrame)
            print(f"Cluster {label}: Closest trace index {indices[mean_idx]} with max distance {filtered_traces[indices[mean_idx]]['distance_um'].max():.2f} μm")
        
        if count_above_threshold > 0:
            cluster_above_threshold[label] = count_above_threshold
            
            
        # --- 4. Formatting for each Subplot ---
        ax.set_xlim(0, t_max)
        ax.axhline(0, color='red', linewidth=0.8, linestyle='--', zorder=12)
        
        # Set the title, handling the noise case
        title = f'Cluster {label} (n={len(indices)})' if label != -1 else f'Noise (n={len(indices)})'
        ax.set_title(title, color=color_map[label] if label != -1 else 'black', fontsize=14)
        
        ax.set_xlabel('Time (s)')
        ax.grid(True, linestyle='--', alpha=0.6)

    # Add a y-axis label only to the first plot in each row for clarity
    for i in range(n_rows):
        axes[i * n_cols].set_ylabel('Distance (μm)')

    # --- 5. Clean Up and Display ---
    plt.suptitle('Trajectory Shapes by Identified Cluster')
    plt.tight_layout() # Adjust layout to make room for suptitle

    plt.savefig(f'result/cluster_img/n_neighbors={n_neighbors}, t_max={t_max}, n_clusters={n_clusters}.png', dpi=300)
    plt.show()

    if closest_trace is not None:
        return cluster_above_threshold, closest_trace
    else:
        return cluster_above_threshold, None










def plot_clustering(embedding,
                 cluster_labels,
                 color_map,
                 t_max,
                 save_path='result/cluster_img/cluster.png'):
    
    fig, ax = plt.subplots(figsize=(10, 7))

    # Create a color palette
    unique_labels = np.unique(cluster_labels)

    # Plot each cluster
    for label in unique_labels:
        indices = np.where(cluster_labels == label)[0]
        if label == -1:
            ax.scatter(embedding[indices, 0], embedding[indices, 1], c='lightgray', s=20, label='Noise')
        else:
            ax.scatter(embedding[indices, 0], embedding[indices, 1], c=[color_map[label]], s=50, label=f'Cluster {label}')

    plt.title(fr'UMAP Projection of Trajectories $t_{{\text{{max}}}}={t_max}$ s, # of clusters={len(np.unique(cluster_labels))}', fontsize=16)
    ax.set_xlabel('UMAP Dimension 1', fontsize=16)
    ax.set_ylabel('UMAP Dimension 2', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
    # plt.legend()
    plt.savefig(save_path, dpi=300)
    plt.show()











def perform_clustering(data_matrix,
                       traces,
                       optimal_k,
                       t_max,
                       n_neighbors=5,
                       plot_cluster=True,
                       plot_rep=True,
                       ):
    """
    Performs UMAP dimensionality reduction followed by KMeans clustering.
    data_matrix: numpy array of shape (n_samples, n_features)
    optimal_k: int, number of clusters to find
    
    Returns:
        cluster_labels: numpy array of shape (n_samples,), cluster assignments
    """
    # --- 1. UMAP: The Mapmaker ---
    # Create the low-dimensional map of your data.
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, n_components=2, random_state=42)
    embedding = reducer.fit_transform(data_matrix)

    # --- 2. KMeans: The Census Taker ---
    # Analyze the map to find a specific number of clusters.
    N_CLUSTERS_TO_FIND = optimal_k  # Use the optimal number of clusters found
    kmeans = KMeans(n_clusters=N_CLUSTERS_TO_FIND, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(embedding)

    # --- 3. Report Results ---
    print(f"Assigned data to {N_CLUSTERS_TO_FIND} clusters using KMeans.")
    for label in np.unique(cluster_labels):
        count = np.sum(cluster_labels == label)
        print(f"Cluster {label}: {count} traces ({count/len(cluster_labels)*100:.1f}%)")

    print(f"Total traces: {len(cluster_labels)}")
    
    # --- Plot 1: The UMAP Projection Colored by Cluster ---
    
    if plot_cluster:
        # Create a color palette
        unique_labels = np.unique(cluster_labels)
        colors = sns.color_palette("deep", len(unique_labels))
        color_map = {label: color for label, color in zip(unique_labels, colors)}

        # Plot the cluster
        plot_clustering(embedding,
                        cluster_labels,
                        color_map,
                        t_max,
                        save_path=f'result/cluster_img/UMAP_n_neighbors={n_neighbors}.png')

        _, cluster_reps =  plot_trajectories(cluster_labels,
                          traces,
                          color_map,
                          t_max,
                          n_neighbors)

    
        return cluster_labels, embedding, cluster_reps
    
    else:
        
        return cluster_labels, embedding, None