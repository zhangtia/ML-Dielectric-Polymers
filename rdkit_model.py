import pandas as pd
# from lolopy.learners import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np
import rdkit.Chem.Descriptors
import rdkit.Chem.Descriptors3D
from rdkit.Chem import MolFromSmiles
import rdkit.Chem.AllChem
import sklearn
from sklearn.model_selection import train_test_split
import forestci
import csv
from itertools import permutations, combinations
from sklearn.metrics import mean_squared_error, r2_score
from get_new_monomers import get_new_SMILES

seed = 98
np.random.seed(seed)

# Get a list of features or descriptors in rdkit
descriptor_strings = ["ExactMolWt",
                      "FpDensityMorgan1",
                      "FpDensityMorgan2",
                      "FpDensityMorgan3",
                      "HeavyAtomMolWt",
                      "MaxAbsPartialCharge",
                      "MaxPartialCharge",
                      "MinAbsPartialCharge",
                      "MinPartialCharge",
                      "MolWt",
                      # "NumRadicalElectrons",
                      "NumValenceElectrons"]

descriptor3D_strings = ["Asphericity",
                        "Eccentricity",
                        "InertialShapeFactor",
                        "NPR1",
                        "NPR2",
                        "PMI1",
                        "PMI2",
                        "PMI3",
                        "RadiusOfGyration",
                        "SpherocityIndex"]



# We need to extract the rdkit class for each feature
strings = [descriptor_strings, descriptor3D_strings]
modules = [rdkit.Chem.Descriptors, rdkit.Chem.Descriptors3D]
rdkit_features = []
feature_strings = []
for feature_names, feature_module in zip(strings, modules):
    for feature_name in feature_names:
        # Get rdkit class from class name
        rdkit_features.append(getattr(feature_module, feature_name))
        feature_strings.append(feature_name)


# # This creates the featurized dataset

# # Read in the unfeaturized dataset
# df = pd.read_csv("4-block-polymer_data_SMILES.csv")
# for feature, feature_name in zip(rdkit_features, feature_strings):
#     def add_feature(row):
#         # This creates a column in the dataset for each of the features
#         if isinstance(row["SMILES"], str):
#             # Make sure the SMILES string can be read
#             if MolFromSmiles(row["SMILES"]) is not None:
#                 # get rdkit molecule
#                 mol = MolFromSmiles(row["SMILES"])
#                 if feature_name in descriptor3D_strings:
#                     # 3D features need the molecules conformation
#                     try:
#                         embed_id = rdkit.Chem.AllChem.EmbedMolecule(mol)
#                         return float(feature(mol, confId=embed_id))
#                     except:
#                         molh = Chem.AddHs(mol) # some features need Hs for it to work
#                         embed_id = rdkit.Chem.AllChem.EmbedMolecule(molh)
#                         return float(feature(molh, confId=embed_id))
#                 else:
#                     return float(feature(mol))
#         return None
#     df[feature_name] = df.apply(add_feature, axis=1) # Create the column
# df.dropna(inplace=True) # Drop blank rows
# print(df)
# df.to_csv("featurized_4-block-polymer_data_SMILES.csv")


# Read in the featurized dataset
df = pd.read_csv("featurized_4-block-polymer_data_SMILES.csv")
print(df)
target1 = "HSE Band Gap (eV)"
target2 = "Total Dielectric Constant"

# Get an array of the featurized values
X_df = df[feature_strings]
X = X_df.values

# Get the target values
y1 = df[target1].values
y2 = df[target2].values

# Split dataset into training and testing
X1_test, X1_train, y1_test, y1_train = train_test_split(X, y1, test_size=0.7, random_state=seed)
X2_test, X2_train, y2_test, y2_train = train_test_split(X, y2, test_size=0.7, random_state=seed)

# Create and train models
model1 = RandomForestRegressor(random_state=seed) # create model
model1.fit(X1_train, y1_train) # fit (train) model
model2 = RandomForestRegressor(random_state=seed) # create model
model2.fit(X2_train, y2_train) # fit (train) model

# Get NDME for band gap model
y1_pred = model1.predict(X1_test)
y1_var = forestci.random_forest_error(model1, X1_train, X1_test)
y1_std = np.sqrt(sum(y1_var) / len(y1_var)) # Get standard deviation
rmse1 = np.sqrt(sum(((y1_test - y1_pred)**2) / len(y1_test)))
ndme1 = rmse1 / y1_std
print(ndme1)

# Get NDME for dielectric constant model
y2_pred = model2.predict(X2_test)
y2_var = forestci.random_forest_error(model2, X2_train, X2_test)
y2_std = np.sqrt(sum(y2_var) / len(y2_var)) # Get standard deviation
rmse2 = np.sqrt(sum(((y2_test - y2_pred)**2) / len(y2_test)))
ndme2 = rmse2 / y2_std
print(ndme2)

# Plot predicted vs actual
plt.figure()
plt.errorbar(y1_test, y1_pred, yerr=np.sqrt(y1_var), fmt='o', color='k', lw=0.5, zorder=0)
plt.scatter(y1_test, y1_pred, color='b', zorder=10)
ylim = plt.gca().get_ylim()
xlim = plt.gca().get_xlim()
plt.plot([0, ylim[1]], [0, ylim[1]], color='g')
plt.xlabel('Actual Band Gap (eV)', fontsize=14)
plt.ylabel('Predicted Band Gap (eV)', fontsize=14)
plt.show()

plt.figure()
plt.errorbar(y2_test, y2_pred, yerr=np.sqrt(y2_var), fmt='o', color='k', lw=0.5, zorder=0)
plt.scatter(y2_test, y2_pred, color='b', zorder=10)
ylim = plt.gca().get_ylim()
xlim = plt.gca().get_xlim()
plt.plot([0, ylim[1]], [0, ylim[1]], color='g')
plt.xlabel('Actual Dielectric Constant', fontsize=14)
plt.ylabel('Predicted Dielectric Constant', fontsize=14)
plt.show()

known_SMILES = list(df["SMILES"]) # Get list of SMILES strings we already have
blocks = sorted(['CH2', 'C6H4', 'NH', 'CO', 'CS', 'C4H2S', 'O'])

new_df = pd.DataFrame() # Create a new dataset of our predicted values
SMILES_and_blocks = get_new_SMILES(known_SMILES, blocks, 4)
new_df["SMILES"] = SMILES_and_blocks[0]
new_df["4-block polymer"] = SMILES_and_blocks[1]

# Featurize the new molecules the same as before
for feature, feature_name in zip(rdkit_features, feature_strings):
    def add_feature(row):
        if isinstance(row["SMILES"], str):
            if MolFromSmiles(row["SMILES"]) is not None:
                mol = MolFromSmiles(row["SMILES"])
                if feature_name in descriptor3D_strings:
                    try:
                        embed_id = rdkit.Chem.AllChem.EmbedMolecule(mol)
                        return float(feature(mol, confId=embed_id))
                    except:
                        molh = rdkit.Chem.AddHs(mol)
                        embed_id = rdkit.Chem.AllChem.EmbedMolecule(molh)
                        return float(feature(molh, confId=embed_id))
                else:
                    return float(feature(mol))
        return None
    new_df[feature_name] = new_df.apply(add_feature, axis=1)

print(new_df)
new_X = new_df[feature_strings].values # Get feature values

# Get predictions for the new monomers
predicted_bandgaps = model1.predict(new_X)
bandgap_var = forestci.random_forest_error(model1, X1_train, new_X)
bandgap_std = np.sqrt(sum(bandgap_var) / len(bandgap_var)) # Get standard deviation

predicted_dielectric_constants = model2.predict(new_X)
dielectric_var = forestci.random_forest_error(model2, X2_train, new_X)
dielectric_std = np.sqrt(sum(dielectric_var) / len(dielectric_var)) # Get standard deviation

new_df["Band Gap (eV)"] = predicted_bandgaps
new_df["Band Gap Uncertainty (eV)"] = np.sqrt(bandgap_var)
new_df["Total Dielectric Constant"] = predicted_dielectric_constants
new_df["Total Dielectric Constant Uncertainty"] = np.sqrt(dielectric_var)

import scipy.stats as stat

def get_probability_of_improvement(row):
    # get Z-scores: (x - mu) / sigma
    Z_bandgap = (5 - row["Band Gap (eV)"]) / row["Band Gap Uncertainty (eV)"]
    Z_dielectric = (5 - row["Total Dielectric Constant"]) / row["Total Dielectric Constant Uncertainty"]
    # get p-values: 1 - area up to x
    p_bandgap = 1 - stat.norm.cdf(Z_bandgap)
    p_dielectric = 1 - stat.norm.cdf(Z_dielectric)
    return p_dielectric * p_bandgap  # return joint probability

# Get probability each monomer meets design goals
new_df["Probability of Improvement"] = new_df.apply(get_probability_of_improvement, axis=1)
new_df.sort_values("Probability of Improvement", axis=0, ascending=False, inplace=True)
print(new_df)
new_df.to_csv("new_4-block_monomers_uncertainty_good_monomers.csv") # Save values



# Plot the pareto front and see if we extended it
plt.figure()
plt.scatter(new_df["Band Gap (eV)"], new_df["Total Dielectric Constant"], color='g')
plt.scatter(df["HSE Band Gap (eV)"], df["Total Dielectric Constant"], color='b')
plt.legend(["Predicted", "Known"])
plt.xlabel("Band Gap (eV)", fontsize=14)
plt.ylabel("Total Dielectric Constant", fontsize=14)
plt.show()

