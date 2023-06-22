import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(color_codes=True, style="white")
pd.options.mode.chained_assignment = None  # default='warn'

# concat all csv files
df_all = pd.read_csv(
    "/Volumes/AnalysisGG/PROCESSED_DATA/Training-ilastik/livecell_100ms-Dcp1a_HOPS/condensates_AIO-pleaserename.csv"
)
os.chdir(
    "/Volumes/AnalysisGG/PROCESSED_DATA/Training-ilastik/livecell_100ms-Dcp1a_HOPS/"
)


#####################################
# Section 1: Condensate Properties
def hist_lineplot(column_name):
    global df_all
    plt.figure(figsize=(6, 4), dpi=300)
    # add cut off by 0.95 quantile
    quantile = np.quantile(df_all[column_name], 0.99)
    ax = sns.kdeplot(
        data=df_all,
        x=column_name,
        common_norm=False,
        clip=(0, quantile),
    )
    plt.xlabel(column_name, weight="bold")
    plt.ylabel("Probability Density", weight="bold")
    fname_save = "Condensate_" + column_name + "_kde.png"
    plt.savefig(fname_save, format="png", bbox_inches="tight")
    plt.close()


hist_lineplot("R_nm")
hist_lineplot("mean_intensity")
hist_lineplot("aspect_ratio")
