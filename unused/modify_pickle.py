import pickle
import os

# Specify the path to the pickle file
folder = "/Volumes/lsa-nwalter/Guoming_Gao_turbo/Walterlab_server/PROCESSED_DATA/RNA-in-HOPS_condensates/paper_figure_plots/PCF"
os.chdir(folder)
for fname_p in [f for f in os.listdir(".") if f.endswith(".p")]:
    # Load the dictionary from the pickle file
    with open(fname_p, "rb") as file:
        data = pickle.load(file)

    # Change the names of the keys
    data["lst_N_loc_condensate"] = data.pop("lst_N_locations_FUS")
    data["lst_N_loc_RNA"] = data.pop("lst_N_locations_RNA")

    # Save the updated dictionary back to the pickle file
    with open(fname_p, "wb") as file:
        pickle.dump(data, file)

    print("Key names updated successfully for", fname_p)
