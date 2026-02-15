
print("Importing libraries...")
import os
import json
import zipfile
import numpy as np
import umap
from sklearn.manifold import trustworthiness
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


data_dir = "./processed_data"
# Iterate over all zip files in the directory
file_list = [f for f in os.listdir(data_dir) if f.endswith('.json.zip')]
file_list.sort()

print(f"Found {len(file_list)} files:")
for idx, filename in enumerate(file_list):
    print(f"    {idx}: {filename}")

selected_files_indices = input("Enter the indices of the files to use (comma separated): ").split(',')
try:
    file_list = [file_list[int(i)] for i in selected_files_indices]
    print(f"Selected file(s): {'.'.join(file_list) if len(file_list) > 1 else file_list[0]}")
except ValueError:
    print("Invalid input. Using all files.")

def load_data():
    all_embeddings = []
    file_list = [f for f in os.listdir(data_dir) if f.endswith('.json.zip')]
    file_list.sort()
        
    for filename in file_list:
        try:
            with zipfile.ZipFile(os.path.join(data_dir, filename), 'r') as z:
                json_name = z.namelist()[0]
                with z.open(json_name) as f:
                    data = json.load(f)
                    traces = data.get('traces', [])
                    for trace in traces:
                        if 'metadata' in trace and 'embedding' in trace['metadata']:
                            emb = trace['metadata']['embedding']
                            if isinstance(emb, dict):
                                emb = [emb[k] for k in sorted(emb.keys(), key=lambda x: int(x))]
                            all_embeddings.append(emb)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            
    embeddings = np.array(all_embeddings, dtype=float)
    valid_mask = np.isfinite(embeddings).all(axis=1)
    return embeddings[valid_mask]

# Load Data
print("Loading data...")
embeddings = load_data()
print(f"Embeddings shape: {embeddings.shape}")

# Define Parameter Grid
n_neighbors_list = np.arange(5, 100, 5)
min_dist_list = np.arange(0.0, 1.0, 0.1)

results = []
best_score = 0

print("Running Grid Search...\n")
for n in n_neighbors_list:
    for d in min_dist_list:
            print(f"Testing n_neighbors={n}, min_dist={d:.1f} ...\n", end="", flush=True)
            
            # Run UMAP with fixed random state for reproducibility
            reducer = umap.UMAP(n_neighbors=n, min_dist=d, n_components=2, random_state=42)
            embedding_2d = reducer.fit_transform(embeddings)
        
            # Calculate Trustworthiness Score (higher is better, max 1.0)
            # This measures how well the local structure is preserved
            score = trustworthiness(embeddings, embedding_2d, n_neighbors=n)
            
            results.append({
                'n_neighbors': n,
                'min_dist': d,
                'trustworthiness': score
            })
            
            # Plot the embedding
            plt.figure(figsize=(8, 8))
            plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], s=1, alpha=0.5)
            plt.title(f"UMAP (n={n}, dist={d})\nTrustworthiness: {score:.4f}")
            plt.axis('off')
            plt.savefig(f"./results/UMAP/tests/combinations/{'_'.join(file_list) if len(file_list) > 1 else file_list[0]}_umap_n{n}_d{d:.1f}.png")
            plt.close()


            # Check if the score is better than the best score
            if score > best_score:
                best_score = score
                print(f"New best score: {best_score:.4f} (n_neighbors={n}, min_dist={d:.1f})\n")
            else:
                print(f"Score: {score:.4f} (n_neighbors={n}, min_dist={d:.1f})\n")
            

# Convert to DataFrame and Sort
df_results = pd.DataFrame(results)
df_results = df_results.sort_values(by='trustworthiness', ascending=False)

print("\n--- Optimized UMAP Parameters (Ranked by Trustworthiness) ---")
print(df_results)

# Save results
df_results.to_csv("umap_tuning_results.csv", index=False)

# Plot the best one
best_params = df_results.iloc[0]
best_n = int(best_params['n_neighbors'])
best_d = best_params['min_dist']

print(f"\nBest Parameters: n_neighbors={best_n}, min_dist={best_d}")
print("Generating plot for best parameters...")

reducer = umap.UMAP(n_neighbors=best_n, min_dist=best_d, n_components=2, random_state=42)
embedding_best = reducer.fit_transform(embeddings)

plt.figure(figsize=(8, 8))
plt.scatter(embedding_best[:, 0], embedding_best[:, 1], s=1, alpha=0.5)
plt.title(f"Best UMAP (n={best_n}, dist={best_d})\nTrustworthiness: {best_params['trustworthiness']:.4f}")
plt.axis('off')
plt.savefig("UMAP/tests/combinations/best_umap_params.png")
print("Saved best_umap_params.png")
