import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np
import forestci
import featurize_monomers
import scipy.stats as stat

seed = 59
feature_strings = featurize_monomers.get_feature_strings()

# Read in the featurized dataset
df = featurize_monomers.get_featurized_dataset("featurized_4-block-polymer_data_SMILES.csv")

# Split monomers into training and testing
# Do sequential learning on testing data
test_i = np.random.choice(list(range(len(df))), int(0.3 * len(df)), replace=False)
test_df = df.iloc[test_i]

# Sort test monomers by band gap * dielectric constant
test_df["Combined"] = test_df[target1] * test_df[target2]
test_df.sort_values("Combined", ascending=False, inplace=True)
test_df.reset_index(inplace=True)

# Store indices of test monomers to track sequential learning performance
test_df["index"] = test_df.index
train_df = df.iloc[[i for i in range(len(df)) if i not in test_i]]

# Get an array of the featurized values
X_test = test_df[feature_strings].values
X_train = train_df[feature_strings].values

y_train = train_df[["HSE Band Gap (eV)", "Total Dielectric Constant"]].values
y_test = test_df[["index", "HSE Band Gap (eV)", "Total Dielectric Constant"]].values

i_list = []
for j in range(len(y_test)):
    print()
    print("Sequential Learning Run #" + str(j))
    print()
    # Create and train models
    model1 = RandomForestRegressor(random_state=seed) # create model
    model1.fit(X_train, y_train[:, 0]) # fit (train) model
    model2 = RandomForestRegressor(random_state=seed) # create model
    model2.fit(X_train, y_train[:, 1]) # fit (train) model

    # Make predictions
    y1_pred = model1.predict(X_test)
    y1_var = forestci.random_forest_error(model1, X_train, X_test)
    y1_std = np.sqrt(sum(y1_var) / len(y1_var)) # Get standard deviation

    y2_pred = model2.predict(X_test)
    y2_var = forestci.random_forest_error(model2, X_train, X_test)
    y2_std = np.sqrt(sum(y2_var) / len(y2_var)) # Get standard deviation


    # get probability of meeting design goals; get Z-scores: (x - mu) / sigma
    Z_bandgap = (5 - y1_pred) / np.sqrt(y1_var)
    Z_dielectric = (5 - y2_pred) / np.sqrt(y2_var)

    # get p-values: 1 - area up to x
    p_bandgap = 1 - stat.norm.cdf(Z_bandgap)
    p_dielectric = 1 - stat.norm.cdf(Z_dielectric)
    probs = p_dielectric * p_bandgap  # joint probability

    # Choose the monomer with the highest probability of meeting our design goals
    besti = int(y_test[np.argmax(probs), 0])
    i_list.append(besti) # Store the rank of the monomer

    # Add the actual DFT values of the monomer to the training data
    X_train = np.append(X_train, X_test[np.abs(y_test[:, 0] - besti) < 1e-1, :].reshape(1, len(feature_strings)), axis=0)
    y_train = np.append(y_train, y_test[np.abs(y_test[:, 0] - besti) < 1e-1, 1:].reshape(1, 2), axis=0)

    # Remove the selected monomer from the testing data
    X_test = X_test[np.abs(y_test[:, 0] - besti) > 1e-1]
    y_test = y_test[np.abs(y_test[:, 0] - besti) > 1e-1]

# Plot the rank of the selected monomer vs sequential learning run number
# Should be linear with slope 1, i.e. best monomer chosen first, worst monomer chosen last
plt.figure()
plt.plot([0, len(i_list)], [0, len(i_list)], color='g', zorder=0)
plt.scatter(list(range(len(i_list))), i_list, color='purple', zorder=100)
plt.legend(['Ideal', 'Actual'])
plt.xlabel("Sequential Learning Run", fontsize=14)
plt.ylabel("Selected Candidate Monomer Rank", fontsize=14)
plt.show()


