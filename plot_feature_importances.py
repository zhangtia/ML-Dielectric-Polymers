import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import sklearn
from sklearn.model_selection import train_test_split
import forestci
import rdkit.Chem.Descriptors
import rdkit.Chem.Descriptors3D
from rdkit.Chem import MolFromSmiles
import rdkit.Chem.AllChem
import numpy as np

seed = 83
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

feature_strings = ["RadiusOfGyration", "InertialShapeFactor", "PMI1", "PMI2", "PMI3", "Asphericity"]
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
# print(r2_score(y1_test, y1_pred))
# Get NDMSE for dielectric constant model
y2_pred = model2.predict(X2_test)
y2_var = forestci.random_forest_error(model2, X2_train, X2_test)
y2_std = np.sqrt(sum(y2_var) / len(y2_var)) # Get standard deviation
rmse2 = np.sqrt(sum(((y2_test - y2_pred)**2) / len(y2_test)))
ndme2 = rmse2 / y2_std
# print(r2_score(y2_test, y2_pred))
print(ndme2)

def bar_plot_horizontal(x, y, n=20):
    """
    Create horizonatal bar plot instance.
    Use 'n' argument to limit the number of bars shown.
    """
    x, y = x[:n], y[:n]
    print(x)
    bars = plt.barh(np.arange(len(x)), y[::-1], height=0.5)
    plt.gca().set_yticks(np.arange(len(x)))
    plt.gca().set_yticklabels(list(x)[::-1])
    plt.gca().set_ylim([-1, len(x)])
    return bars[::-1]


# get feature importances
feature_importances1 = model1.feature_importances_
feature_importances2 = model2.feature_importances_

plt.figure()
plt.barh(np.arange(10), feature_importances1[:10][::-1])
plt.yticks(ticks=np.arange(10), labels=feature_strings[:10][::-1])
plt.ylim([-1, 10])
plt.show()

plt.figure()
plt.barh(np.arange(10), feature_importances2[:10][::-1])
plt.yticks(ticks=np.arange(10), labels=feature_strings[:10][::-1])
plt.ylim([-1, 10])
plt.show()

