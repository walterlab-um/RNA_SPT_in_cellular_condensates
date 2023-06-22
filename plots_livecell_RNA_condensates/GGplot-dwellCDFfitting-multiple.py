import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from copy import deepcopy
from scipy.optimize import curve_fit
from lmfit import Minimizer, Parameters, report_fit
import seaborn as sns

sns.set(color_codes=True, style="white")

###############################################
# Prepare non linear fitting prerequisites
def CDF(t, a1, a2, k1, k2):
    return 1 - a1 * np.exp(-k1 * t) - a2 * np.exp(-k2 * t)


def calc_R2(ydata, yfit):
    # residual sum of squares (ss_tot)
    residuals = ydata - yfit
    ss_res = np.sum(residuals ** 2)
    # total sum of squares (ss_tot)
    ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
    # r_squared-value
    r_squared = 1 - (ss_res / ss_tot)

    return r_squared


# define objective function: returns the array to be minimized
def CDF_lmfit(params, t, data):
    """Model a decaying sine wave and subtract data."""
    a1 = params["a1"]
    a2 = params["a2"]
    k1 = params["k1"]
    k2 = params["k2"]
    model = 1 - a1 * np.exp(-k1 * t) - a2 * np.exp(-k2 * t)
    return model - data


# create a set of Parameters
params = Parameters()
params.add("a1", min=0, max=1, value=0.8)
params.add("a2", expr="1-a1")
params.add("k1", min=0.03, max=7, value=5)  # min=ln2/0.1
params.add("k2", min=0.03, max=7, value=0.1)


###############################################
# Main

path = "/Volumes/AnalysisGG/PROCESSED DATA/2022May-LiveCellCleanup"


lst_RNA = ["miR21", "THOR", "THORdel", "FL", "FL_wCHX", "miRcxcr4"]
lst_files = [f + "_2x-dwelltimes.csv" for f in lst_RNA]

lst_percent1 = []
lst_percent2 = []
lst_halflife1 = []
lst_halflife2 = []
lst_R2 = []
lst_percent1_std = []
lst_percent2_std = []
lst_halflife1_std = []
lst_halflife2_std = []
os.chdir(path)
for i in range(len(lst_RNA)):
    plt.figure(figsize=(9, 3), dpi=300)
    data = pd.read_csv(lst_files[i])["dwelltimes"].to_numpy() * 0.1

    hist, bin_edges, _ = plt.hist(
        data,
        bins=100,
        range=(0, 20),
        density=True,
        histtype="stepfilled",
        cumulative=1,
        color="gray",
        label=lst_RNA[i],
        alpha=0.5,
    )

    bin_centers = bin_edges[:-1] + 0.1
    # do fit, with the default leastsq algorithm
    minner = Minimizer(CDF_lmfit, params, fcn_args=(bin_centers, hist))
    result = minner.minimize()
    print(lst_RNA[i])
    report_fit(result)

    a1 = result.params["a1"].value
    a2 = result.params["a2"].value
    k1 = result.params["k1"].value
    k2 = result.params["k2"].value
    a1_std = result.params["a1"].stderr
    a2_std = result.params["a2"].stderr
    k1_std = result.params["k1"].stderr
    k2_std = result.params["k2"].stderr

    xdata = bin_centers
    ydata = hist
    yfit = hist + result.residual
    R2 = calc_R2(ydata, yfit)

    x = np.arange(0, 20, 0.05)
    plt.plot(bin_centers, yfit, color=sns.color_palette()[i], linestyle="-")
    # Assuming first order reaction, also note in python np.log is ln
    halflife1 = k1
    halflife2 = k2
    halflife1_std = k1_std
    halflife2_std = k2_std

    plt.legend(loc="lower right")
    plt.xlabel("Dwell Time (s)")
    plt.ylabel("CDF")
    plt.xlim(0, 20)
    plt.ylim(0, 1)
    plt.tight_layout()

    comment = (
        "Dwelling Half-Life[1]: "
        + str(round(halflife1, 2))
        + " s, "
        + str(round(a1 * 100, 2))
        + "%"
    )
    plt.text(
        19.8, 0.8, comment, horizontalalignment="right", size=20, fontweight="normal"
    )
    comment = (
        "Dwelling Half-Life[2]: "
        + str(round(halflife2, 2))
        + " s, "
        + str(round(a2 * 100, 2))
        + "%"
    )
    plt.text(
        19.8, 0.65, comment, horizontalalignment="right", size=20, fontweight="normal"
    )
    comment = "R^2 = " + str(round(R2, 3))
    plt.text(
        19.8, 0.45, comment, horizontalalignment="right", size=25, fontweight="bold"
    )

    fname_save = "Dwell Time Distribution-" + lst_RNA[i] + "-CDF-double_fit.svg"
    plt.savefig(fname_save, format="svg")
    plt.close()

    lst_percent1.append(a1 * 100)
    lst_percent2.append(a2 * 100)
    lst_halflife1.append(halflife1)
    lst_halflife2.append(halflife2)
    lst_R2.append(R2)
    lst_percent1_std.append(a1_std * 100)
    lst_percent2_std.append(a2_std * 100)
    lst_halflife1_std.append(halflife1_std)
    lst_halflife2_std.append(halflife2_std)

df_save = pd.DataFrame(
    {
        "RNA": np.hstack(lst_RNA),
        "percent1": np.hstack(lst_percent1),
        "percent1_std": np.hstack(lst_percent1_std),
        "percent2": np.hstack(lst_percent2),
        "percent2_std": np.hstack(lst_percent2_std),
        "halflife1": np.hstack(lst_halflife1),
        "halflife1_std": np.hstack(lst_halflife1_std),
        "halflife2": np.hstack(lst_halflife2),
        "halflife2_std": np.hstack(lst_halflife2_std),
        "R2": np.hstack(lst_R2),
    }
)

df_save.to_csv("Dwell Time fitting results_double.csv", index=False)
