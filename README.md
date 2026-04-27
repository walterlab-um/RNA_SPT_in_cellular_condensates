# RNA_SPT_in_cellular_condensates
Live-cell RNA single molecule tracking within condensates, such as P bodies (PB), stress granules (SG), or hyperosmotic phase separation (HOPS) condensates by a variety of proteins.


# Key notebooks (Sam's edits)

## Tracking distance to condensate

For this, the data folder (`data/`) is too big to upload to GitHub, so you may need to put it in a folder manually from Turbo. The structure should be like this:

```
data/
├── THOR_1x/
│   └── condensate/
│       ├── condensates_AIO-20220409-THOR_1x_FOV_1-cropped-left/
│       └── .../
│   └── RNA/
│       ├── SPT_results_AIO-20220409-THOR_1x_FOV_1-right_reformatted/
│       └── .../
├── THOR_2x/
│   └── condensate/
│       ├── condensates_AIO-20220528-THOR_2x_FOV_1-cropped-left/
│       └── .../
│   └── RNA/
│       ├── SPT_results_AIO-20220528-THOR_2x_FOV_1-right_reformatted/
│       └── .../
└── ...
```

- **Tracks with Condesate (Debugging)**: `Tracks-with-condensate_debugging.ipynb`
    - This is the main notebook for analysis of the SPT data.
    - Calculate the shortest distance from each trajectory to the nearest condensate boundary using Shapely.
    - Since the condensate contour are sometimes buggy (refer to `media/snapshot_20220508-FLmRNA_2x_FOV-17_track_38.mp4`), the results are filtered based on the consistency of condensate contour position across frames.
    - Results are saved to `result` folder; for filtered, it has the prefix `{data_name}_filtered_rna_condensate_distances.csv` and for each experiment that are saved individually, it has the prefix `{exp}.csv`.
    - **Note**: I believe the code should be stable enough such that you can just run it directly by selecting the dataset. Please let me know if it doesn't work though.

- **Tracks with Condesate (BPS ver.)**: `Tracks-with-condensate_bps.ipynb`
    - This notebook is mainly used to plot representative trajectories for a chosen data set for BPS 2026.
    - Calculate the shortest distance from each trajectory to the nearest condensate boundary.
    - Plot the representative trajectory(ies) for a chosen data set.

## META-SiM and clustering

- **META-SiM Analysis**: `META-SiM/META_SiM_Analysis.ipynb`
    - Analysis of the META-SiM data from Leo and Jieming, the folder for the processed data is here: `META-SiM/processed_data/`.
    - UMAP clustering and processing trajectories by clusters.
    - Results are saved in `META-SiM/results/`.
        - UMAP cluster result: `META-SiM/results/clustering_results/`. Contains the cluster indice and the trajectory indices.
        - UMAP cluster plots: `META-SiM/results/UMAP/`. Contains the UMAP plots for each cluster.
        - Traces (**unused**): `META-SiM/results/traces/`. Contains the representative (mean) of the trajectory from each cluster.
    - Note: there's a raw data folder (`META-SiM/raw_data/`) but its not used, it was made in the beginning when I tried to process the raw data myself.

- **Post Clustering Analysis**: `META-SiM/Post_clustering_analysis.ipynb`
    - Population analysis of the clustering results from `META-SiM/META_SiM_Analysis.ipynb`. The code was written to compare two or more clusters.
    - Results are saved in `META-SiM/results/`.
        - Population comparison: `META-SiM/results/population_comparison/`. Contains plots of population comparison between clusters.

# Ignore these folders (development archives):
- `clustering/...`