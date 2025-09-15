import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.collections import LineCollection

sns.set(style="white", context="talk")

# ───────────────────────────────────────────────────
# CONFIGURATION
# ───────────────────────────────────────────────────
folder = "/Users/esumrall/Desktop/RNA-in-HOPS_condensates/FL_2x_100ms/colocalization" # update
os.chdir(folder)

fn_list = [
    # ("colocalization_AIO_concat-FL_1x_100ms_original.csv", "1x"),
    # ("colocalization_AIO_concat-FL_2x_100ms_original.csv", "2x"),
    ("colocalization_AIO-20220508-FLmRNA_2x_FOV-21-cropped-left_original.csv", "2x")
]
condition2color = {"2x": "#E88791"} #, "2x": "#BD2433"}

def nm_to_um(x_nm): return x_nm / 1000.0

# ───────────────────────────────────────────────────
# LOAD AND CONCATENATE DATA
# ───────────────────────────────────────────────────
dfs = []
for fn, cond in fn_list:
    df = pd.read_csv(fn)
    df["distance_um"] = nm_to_um(df["distance_to_edge_nm"].fillna(0))
    df["condition"] = cond
    dfs.append(df)
df_all = pd.concat(dfs, ignore_index=True)

# ───────────────────────────────────────────────────
# EXTRACT DWELL SEGMENTS AND PREP TIME-ALIGNED DISTANCE TRACES
# ───────────────────────────────────────────────────
traces = []  # list of (trackID, condition, times, distances)
for (track_id, cond), df_track in df_all.groupby(["RNA_trackID", "condition"]):
    df_track = df_track.sort_values("t")
    condarr = df_track["InCondensate"].to_numpy()
    tvals = df_track["t"].to_numpy() * 0.1  # if t is frame, *0.1s
    dist = df_track["distance_um"].to_numpy()
    change_idx = np.where(condarr[:-1] != condarr[1:])[0] + 1
    idxs = np.concatenate(([0], change_idx, [len(condarr)]))
    for i in range(len(idxs)-1):
        start, end = idxs[i], idxs[i+1]
        if condarr[start]:  # dwell
            times = tvals[start:end] - tvals[start]
            distances = dist[start:end]
            if len(times) > 1:
                traces.append((track_id, cond, times, distances))

# ───────────────────────────────────────────────────
# OVERLAID PLOT, COLORED BY CONDITION
# ───────────────────────────────────────────────────
plt.figure(figsize=(8, 6))
ax = plt.gca()
for track_id, cond, times, distances in traces:
    ax.plot(times, distances, color=condition2color[cond], alpha=0.26, lw=1.5)
# Population median per condition
all_times_uniform = np.linspace(0, max([t.max() for _,_,t,_ in traces if len(t)>0]), 100)
for cond, color in condition2color.items():
    all_distances_interp = []
    for _, c, times, distances in traces:
        if c != cond: continue
        if len(times) > 1:
            interp = np.interp(all_times_uniform, times, distances, left=np.nan, right=np.nan)
            all_distances_interp.append(interp)
    if all_distances_interp:
        med_trace = np.nanmedian(np.stack(all_distances_interp), axis=0)
        ax.plot(all_times_uniform, med_trace, color=color, lw=3, label=f"{cond} median")
ax.set_xlim(0, np.max([times.max() for _, _, times, _ in traces]))
ax.set_ylim(0, np.max([dist.max() for _, _, _, dist in traces])*1.05)
ax.set_xlabel("Time since first dwell (s)")
ax.set_ylabel("Distance from boundary (μm)")
ax.set_title("Distance vs Time, Colored by Condition")
ax.legend(frameon=False)
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("distance_vs_time_per_track_both_colors.png", dpi=300)

# ───────────────────────────────────────────────────
# SPLIT PLOTS BY ESCAPE/BOUND WITH CONDITION COLORS
# ───────────────────────────────────────────────────
escape_threshold = 0.245  # μm, adjust if needed
escape_traces, bound_traces = [], []
for track_id, cond, times, distances in traces:
    if len(distances) == 0:
        continue
    if np.max(distances) > escape_threshold:
        escape_traces.append((track_id, cond, times, distances))
    else:
        bound_traces.append((track_id, cond, times, distances))
groups = [("Bound", bound_traces), ("Escape", escape_traces)]

fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True, sharex=True)
max_time = max([times.max() for _,_,times,_ in traces if len(times) > 0])
for ax, (title, group_traces) in zip(axes, groups):
    for track_id, cond, times, distances in group_traces:
        ax.plot(times, distances,
                color=condition2color[cond], alpha=0.36 if title=="Bound" else 0.7, lw=1.5 if title=="Bound" else 2)
    # Median for each condition
    for cond, color in condition2color.items():
        all_dists_interp = []
        for _, c, times, distances in group_traces:
            if c != cond: continue
            if len(times) > 1:
                interp = np.interp(all_times_uniform, times, distances, left=np.nan, right=np.nan)
                all_dists_interp.append(interp)
        if all_dists_interp:
            med_trace = np.nanmedian(np.stack(all_dists_interp), axis=0)
            ax.plot(all_times_uniform, med_trace, color=color, lw=3, label=f"{cond} median")
    ax.set_title(f"{title}\nN={len(group_traces)}")
    ax.set_xlabel("Time since first dwell (s)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=False)
axes[0].set_ylabel("Distance from boundary (μm)")
axes[0].set_xlim(0, max_time)
axes[0].set_ylim(0, np.max([dist.max() for _, _, _, dist in traces])*1.05)
plt.suptitle("Escape vs Bound Trajectories by Condition")
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("distance_vs_time_per_track_facet_by_condition.png", dpi=300)
