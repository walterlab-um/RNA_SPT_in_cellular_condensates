# %--- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Clustering analysis utilities
import umap
import hdbscan
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score



################################################
# Optimizations
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