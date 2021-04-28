import pandas as pd
import matplotlib.pyplot as plt
from itertools import permutations

# Get the monomers found in the original dataset
old_df = pd.read_csv("featurized_4-block-polymer_data_SMILES.csv")

# Get a list of adjacent blocks in each monomer
bonds = []
for i, row in old_df.iterrows():
    row_blocks = list(row["4-block polymer"].split('-'))
    row_bonds = []
    for i in range(len(row_blocks)):
        j = i + 1
        if j >= 4:
            j -= 4
        row_bonds.append(tuple(list(sorted([row_blocks[i], row_blocks[j]]))))
    # Sort adjacent blocks to get unique monomer
    sorted_row_bonds = tuple(list(sorted(row_bonds)))
    if sorted_row_bonds not in bonds:
        bonds.append(sorted_row_bonds)

def is_new(row):
    # Find if monomer is present in original dataset
    row_blocks = list(row["4-block polymer"].split('-'))
    row_bonds = []
    for i in range(len(row_blocks)):
        j = i + 1
        if j >= 4:
            j -= 4
        row_bonds.append(tuple(list(sorted([row_blocks[i], row_blocks[j]]))))
    sorted_row_bonds = tuple(list(sorted(row_bonds)))
    if sorted_row_bonds not in bonds:
        return True
    else:
        return False


new_df = pd.read_csv("new_4-block_monomers.csv")
new_df["New"] = new_df.apply(is_new, axis=1)
# new_df = new_df[new_df["New"] == True]
new_df = new_df[["SMILES", "Total Dielectric Constant", "Total Dielectric Constant Uncertainty", "Band Gap (eV)", "Band Gap Uncertainty (eV)"]]


# Get high uncertainty rating: prediction + uncertainty
new_df["Band Gap (eV) High Uncertainty"] = new_df["Band Gap (eV)"] + new_df["Band Gap Uncertainty (eV)"]
new_df["Total Dielectric Constant High Uncertainty"] = new_df["Total Dielectric Constant"] + new_df["Total Dielectric Constant Uncertainty"]
# Multiply band gap by dielectric constant
new_df["Combined High Uncertainty"] = new_df["Band Gap (eV) High Uncertainty"] * new_df["Total Dielectric Constant High Uncertainty"]
new_df.sort_values("Combined High Uncertainty", axis=0, ascending=False, inplace=True)
print(new_df)

# Plot the best candidates from the high uncertainty rating
plt.figure()
plt.scatter(new_df.iloc[3:]["Band Gap (eV)"], new_df.iloc[3:]["Total Dielectric Constant"], color='g', zorder=10, s=50, alpha=0.75)
plt.scatter(old_df["HSE Band Gap (eV)"], old_df["Total Dielectric Constant"], color='b', zorder=20, s=50, alpha=0.75)
plt.legend(["Predicted", "Known"])
plt.scatter(new_df.iloc[:3]["Band Gap (eV)"], new_df.iloc[:3]["Total Dielectric Constant"], color='g', edgecolors='r', zorder=30, s=70, linewidth=1.75)
xlim = plt.gca().get_xlim()
ylim = plt.gca().get_ylim()
plt.plot([5, 5], [5, 50], color='k', ls='--', lw=0.75, zorder=0)
plt.plot([5, 50], [5, 5], color='k', ls='--', lw=0.75, zorder=0)
plt.xlim(xlim)
plt.ylim(ylim)
plt.xlabel("Band Gap (eV)", fontsize=16)
plt.ylabel("Total Dielectric Constant", fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# Get low uncertainty rating: prediction + uncertainty
new_df["Band Gap (eV) Low Uncertainty"] = new_df["Band Gap (eV)"] - new_df["Band Gap Uncertainty (eV)"]
new_df["Total Dielectric Constant Low Uncertainty"] = new_df["Total Dielectric Constant"] - new_df["Total Dielectric Constant Uncertainty"]
# Multiply band gap by dielectric constant
new_df["Combined Low Uncertainty"] = new_df["Band Gap (eV) Low Uncertainty"] * new_df["Total Dielectric Constant Low Uncertainty"]
new_df.sort_values("Combined Low Uncertainty", axis=0, ascending=False, inplace=True)
print(new_df)

# Plot the best candidates from the low uncertainty rating
plt.figure()
plt.scatter(new_df["Band Gap (eV)"], new_df["Total Dielectric Constant"], color='g', zorder=10, s=50, alpha=0.75)
plt.scatter(old_df["HSE Band Gap (eV)"], old_df["Total Dielectric Constant"], color='b', zorder=20, s=50, alpha=0.75)
plt.legend(["Predicted", "Known"])
plt.scatter(new_df.iloc[[4, 6, 7]]["Band Gap (eV)"], new_df.iloc[[4, 6, 7]]["Total Dielectric Constant"], color='g', edgecolors='r', zorder=30, s=70, linewidth=1.75)
xlim = plt.gca().get_xlim()
ylim = plt.gca().get_ylim()
plt.plot([5, 5], [5, 50], color='k', ls='--', lw=0.75, zorder=0)
plt.plot([5, 50], [5, 5], color='k', ls='--', lw=0.75, zorder=0)
plt.xlim(xlim)
plt.ylim(ylim)
plt.xlabel("Band Gap (eV)", fontsize=16)
plt.ylabel("Total Dielectric Constant", fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
