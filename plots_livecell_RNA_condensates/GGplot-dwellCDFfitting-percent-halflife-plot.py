import os
from os.path import join
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from copy import deepcopy
from scipy.optimize import curve_fit
import seaborn as sns

sns.set(color_codes=True, style="white")

###############################################
# Loading Data
path = "/Volumes/AnalysisGG/PROCESSED DATA/2022May-LiveCellCleanup"
fpath_double = join(path, "Dwell Time fitting results_double.csv")
fpath_single = join(path, "Dwell Time fitting results_single.csv")

df_double = pd.read_csv(fpath_double)
df_single = pd.read_csv(fpath_single)

# lst_RNA = ["miR21", "THOR", "FL"]
# lst_label = ["miRNA", "lncRNA", "mRNA"]
# lst_color = ["gray", "firebrick", "tab:blue"]
# lst_RNA = ["THOR", "THORdel"]
# lst_label = ["THOR", "THOR\u0394"]
# lst_color = ["firebrick", "gray"]
lst_RNA = ["FL", "FL_wCHX"]
lst_label = ["-CHX", "+CHX"]
lst_color = ["tab:blue", "gray"]


os.chdir(path)
plt.figure(figsize=(5, 5), dpi=200)
for i in range(len(lst_RNA)):
    row = df_single[df_single.RNA == lst_RNA[i]]
    if row.R2.to_numpy(dtype=float)[0] > 0.95:
        plt.scatter(
            100, row.halflife, color=lst_color[i], s=50, label=lst_label[i], alpha=0.5
        )
    else:
        row = df_double[df_double.RNA == lst_RNA[i]]
        plt.scatter(
            [row.percent1, row.percent2],
            [row.halflife1, row.halflife2],
            color=lst_color[i],
            s=200,
            label=lst_label[i],
            edgecolors="None",
            alpha=0.5,
        )
        plt.errorbar(
            x=row.percent1,
            y=row.halflife1,
            xerr=row.percent1_std,
            yerr=row.halflife1_std,
            color=lst_color[i],
            capsize=5,
            alpha=0.7,
        )
        plt.errorbar(
            x=row.percent2,
            y=row.halflife2,
            xerr=row.percent2_std,
            yerr=row.halflife2_std,
            color=lst_color[i],
            capsize=5,
            alpha=0.7,
        )


ax = plt.gca()
# ax.set_xscale("log")
# ax.set_yscale("log")
plt.xlabel("Slow-Dwelling Population (%)", weight="bold", fontsize=20)
plt.ylabel("Dwell Rate Constant (s$^{-1}$)", weight="bold", fontsize=20)

# plt.xlim(0, 20)
# plt.ylim(0, 1)
# fname_save = "Dwelling Half-Life distribution among RNAs-1.svg"
# plt.savefig(fname_save, format="svg")

# plt.xlim(80, 100)
# plt.ylim(6, 8)
ax = plt.gca()
ax.tick_params(axis="both", labelsize=20)
# compare three class
# plt.xlim(7, 18)
# plt.ylim(0.4, 0.8)
# THOR
# plt.xlim(5, 10)
# plt.ylim(0.3, 0.7)
# FL
plt.xlim(7, 14)
plt.ylim(0.4, 0.7)

plt.legend(loc=4, frameon=False, prop={"weight": "bold", "size": 17})
plt.tight_layout()
# fname_save = "Dwelling Half-Life distribution among RNAs-compare3class.svg"
# fname_save = "Dwelling Half-Life distribution among RNAs-THORvsdel.svg"
fname_save = "Dwelling Half-Life distribution among RNAs-FLvsFLchx.svg"
plt.savefig(fname_save, format="svg")
plt.close()
