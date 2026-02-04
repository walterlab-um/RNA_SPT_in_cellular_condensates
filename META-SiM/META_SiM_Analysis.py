#!/usr/bin/env python
# coding: utf-8

# # META-SiM Analysis
# 
# This notebook uses the `metasim` and `openfret` libraries to visualize processed single-molecule data.

# In[1]:


# Install dependencies if not already present
# %pip install -q metasim openfret


# In[2]:


import os
import json
import zipfile
import numpy as np
import openfret
import metasim
import matplotlib.pyplot as plt
from tqdm import tqdm


# ## 1. Load Processed Data
# 
# We load the META-SiM processed `.json.zip` files which contain the latent embeddings for each trace.

# In[3]:


data_dir = "./processed_data"

all_embeddings = []
all_labels = []

# Iterate over all zip files in the directory
file_list = [f for f in os.listdir(data_dir) if f.endswith('.json.zip')]
file_list.sort()

print(f"Found {len(file_list)} files: {file_list}")

# Change 'description' to 'desc' to fix TqdmKeyError
for filename in tqdm(file_list, desc="Loading Data"):
    file_path = os.path.join(data_dir, filename)
    
    try:
        with zipfile.ZipFile(file_path, 'r') as z:
            # Assume there is one json file per zip
            json_name = z.namelist()[0]
            with z.open(json_name) as f:
                data = json.load(f)
                
                # Extract traces
                traces = data.get('traces', [])
                # Use the filename (without extension) as the condition label
                condition_name = filename.replace('.json.zip', '')
                
                for trace in traces:
                    if 'metadata' in trace and 'embedding' in trace['metadata']:
                        emb = trace['metadata']['embedding']
                        
                        # Handle case where embedding is stored as a dictionary with string indices
                        if isinstance(emb, dict):
                            # Sort by integer value of the keys
                            sorted_keys = sorted(emb.keys(), key=lambda x: int(x))
                            emb = [emb[k] for k in sorted_keys]
                            
                        all_embeddings.append(emb)
                        all_labels.append(condition_name)
                        
    except Exception as e:
        print(f"Error reading {filename}: {e}")

# Convert lists to numpy arrays safely
try:
    embeddings_raw = np.array(all_embeddings, dtype=float)
except ValueError as e:
    print(f"Error creating numpy array: {e}")
    # Attempt to handle jagged arrays by filtering for the most common length
    lengths = [len(e) for e in all_embeddings]
    mode_len = max(set(lengths), key=lengths.count)
    print(f"Filtering traces to match most common embedding length: {mode_len}")
    
    valid_indices = [i for i, l in enumerate(lengths) if l == mode_len]
    embeddings_raw = np.array([all_embeddings[i] for i in valid_indices], dtype=float)
    labels_raw = np.array([all_labels[i] for i in valid_indices])
else:
    labels_raw = np.array(all_labels)

# 3. Clean Data (Remove NaNs/Infs)
# Identify valid indices where no element is NaN or Inf
valid_mask = np.isfinite(embeddings_raw).all(axis=1)
n_removed = len(embeddings_raw) - np.sum(valid_mask)

if n_removed > 0:
    print(f"⚠️ Removed {n_removed} traces containing NaNs or Infs.")

embeddings = embeddings_raw[valid_mask]
labels = labels_raw[valid_mask]

print(f"\nTotal Valid Traces: {embeddings.shape[0]}")
print(f"Embedding Shape: {embeddings.shape}")


# ## 2. Visualize with META-SiM
# 
# We use the `metasim.fret.tools.viz` module to generate the UMAP projection and plot it.

# In[4]:


# 1. Calculate UMAP reduction
print("Calculating UMAP using metasim tools...")
reducer = metasim.fret.tools.viz.get_umap_reducer(embeddings)
umap_coord = reducer.transform(embeddings)

# 2. Plot UMAP
print("Generating Plot...")
plt.figure(figsize=(12, 10))

# Using plot_umap from metasim
# We pass 'labels' to color/group by experimental condition
metasim.fret.tools.viz.plot_umap(
    umap_coord=umap_coord,
    label=labels
)

plt.title("META-SiM UMAP: Experimental Conditions", fontsize=16, fontweight='bold')
plt.savefig("metasim_umap_viz.png", dpi=300, bbox_inches='tight')
plt.show()

