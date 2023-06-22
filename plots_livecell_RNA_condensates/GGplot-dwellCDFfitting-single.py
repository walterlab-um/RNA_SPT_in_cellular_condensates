import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from copy import deepcopy
from scipy.optimize import curve_fit
import seaborn as sns

sns.set(color_codes=True, style="white")

###############################################
# functions
def CDF(t, k):
    return 1 - np.exp(-k * t)


def calc_R2(ydata, yfit):
    # residual sum of squares (ss_tot)
    residuals = ydata - yfit
    ss_res = np.sum(residuals ** 2)
    # total sum of squares (ss_tot)
    ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
    # r_squared-value
    r_squared = 1 - (ss_res / ss_tot)

    return r_squared


path = "/Volumes/AnalysisGG/PROCESSED DATA/2022May-LiveCellCleanup"
lst_files = [f for f in os.listdir(path) if f.endswith("dwelltimes.csv")]

lst_RNAs = []
lst_halflife = []
lst_R2 = []

os.chdir(path)
for file in lst_files:
    plt.figure(figsize=(9, 3), dpi=300)
    data = pd.read_csv(file)["dwelltimes"].to_numpy() * 0.1

    hist, bin_edges, _ = plt.hist(
        data,
        bins=100,
        range=(0, 20),
        density=True,
        histtype="stepfilled",
        cumulative=1,
        color="gray",
        label=file.split("_2x")[0],
        alpha=0.5,
    )

    bin_centers = bin_edges[:-1] + 0.1
    k_fit, _ = curve_fit(CDF, bin_centers, hist)
    xdata = bin_centers
    ydata = hist
    yfit = CDF(xdata, k_fit[0])
    R2 = calc_R2(ydata, yfit)

    plt.plot(
        np.arange(0, 20, 0.05),
        1 - np.exp(-k_fit[0] * np.arange(0, 20, 0.05)),
        color="gray",
        linestyle="-",
    )
    # Assuming first order reaction, also note in python np.log is ln
    halflife = np.log(2) / k_fit[0]

    plt.legend(loc="lower right")
    plt.xlabel("Dwell Time (s)")
    plt.ylabel("CDF")
    plt.xlim(0, 20)
    plt.tight_layout()

    comment = "Dwelling Half-Life: " + str(round(halflife, 2)) + " s"
    plt.text(
        19.8, 0.8, comment, horizontalalignment="right", size=20, fontweight="normal"
    )

    comment = "R^2 = " + str(round(R2, 3))
    plt.text(
        19.8, 0.6, comment, horizontalalignment="right", size=25, fontweight="bold"
    )

    fname_save = (
        "Dwell Time Distribution-" + file.split("_2x")[0] + "-CDF-single_fit.svg"
    )
    plt.savefig(fname_save, format="svg")
    plt.close()

    lst_RNAs.append(file.split("_2x")[0])
    lst_halflife.append(halflife)
    lst_R2.append(R2)

df_save = pd.DataFrame(
    {
        "RNA": np.hstack(lst_RNAs),
        "halflife": np.hstack(lst_halflife),
        "R2": np.hstack(lst_R2),
    }
)
df_save.to_csv("Dwell Time fitting results_single.csv", index=False)
