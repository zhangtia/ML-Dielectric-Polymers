import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import forestci
from get_new_monomers import get_new_SMILES
import featurize_monomers
import scipy.stats as stats
from collections import namedtuple

# Create a namedtuple to store the model and ndme score
ModelAndScore = namedtuple("ModelAndScore", ['model', 'score'])

seed = 59
np.random.seed(seed)

def get_models(plot=True):
    """ Get the band gap and dielectric constant models.

        Args:
            plot (bool): Whether or not to plot predicted vs actual and feature importance plots

        Returns:
            (band_gap, dielectric_constant): tuple of two ModelAndScore namedtuples, one holding
            the band gap model and ndme and one hold the dielectric constant model and ndme
    """
    feature_strings = featurize_monomers.get_feature_strings()

    # Read in the featurized dataset
    df = featurize_monomers.get_featurized_dataset("featurized_4-block-polymer_data_SMILES.csv")

    # Get an array of the feature values for each monomer
    feature_values = df[feature_strings].values

    # We are trying to predict band gap and dielectric constant
    band_gap = df["HSE Band Gap (eV)"].values
    dielectric_constant = df["Total Dielectric Constant"].values


    # Split the data into train and test data
    # Get indices for the test and train data
    test_i = np.random.randint(0, len(feature_values), int(0.3 * len(feature_values)))
    train_i = np.asarray([i for i in range(len(feature_values)) if i not in test_i])

    # Get test and train data values from indices
    feature_values_test = feature_values[test_i]
    feature_values_train = feature_values[train_i]
    band_gap_test = band_gap[test_i]
    dielectric_constant_train = dielectric_constant[train_i]
    dielectric_constant_test = dielectric_constant[test_i]
    band_gap_train = band_gap[train_i]

    # Split dataset into training and testing
    # feature_values1_test, feature_values1_train, band_gap_test, band_gap_train = train_test_split(feature_values, band_gap, test_size=0.7, random_state=seed)
    # feature_values2_test, feature_values2_train, dielectric_constant_test, dielectric_constant_train = train_test_split(feature_values, dielectric_constant, test_size=0.7, random_state=seed)

    # Create and train models
    band_gap_model = RandomForestRegressor(random_state=seed) # create model
    band_gap_model.fit(feature_values_train, band_gap_train) # fit (train) model
    dielectric_constant_model = RandomForestRegressor(random_state=seed) # create model
    dielectric_constant_model.fit(feature_values_train, dielectric_constant_train) # fit (train) model

    # Get NDME for band gap model
    band_gap_pred = band_gap_model.predict(feature_values_test)
    band_gap_var = forestci.random_forest_error(band_gap_model, feature_values_train, feature_values_test)
    band_gap_std = np.sqrt(sum(band_gap_var) / len(band_gap_var)) # Get standard deviation
    band_gap_normalized_error = (band_gap_pred - band_gap_test) / band_gap_std
    band_gap_rmse = np.sqrt(sum(((band_gap_test - band_gap_pred)**2) / len(band_gap_test)))
    band_gap_ndme = band_gap_rmse / np.std(band_gap_test)
    print("NDME for band gap model: " + str(round(band_gap_ndme, 3)))
    band_gap = ModelAndScore(model=band_gap_model, score=band_gap_ndme)

    # Get NDME for dielectric constant model
    dielectric_constant_pred = dielectric_constant_model.predict(feature_values_test)
    dielectric_constant_var = forestci.random_forest_error(dielectric_constant_model, feature_values_train, feature_values_test)
    dielectric_constant_std = np.sqrt(sum(dielectric_constant_var) / len(dielectric_constant_var)) # Get standard deviation
    dielectric_constant_normalized_error = (dielectric_constant_pred - dielectric_constant_test) / dielectric_constant_std
    dielectric_constant_rmse = np.sqrt(sum(((dielectric_constant_test - dielectric_constant_pred)**2) / len(dielectric_constant_test)))
    dielectric_constant_ndme = dielectric_constant_rmse / np.std(dielectric_constant_test)
    print("NDME for dielectric constant model: " + str(round(dielectric_constant_ndme, 3)))
    dielectric_constant = ModelAndScore(model=dielectric_constant_model, score=dielectric_constant_ndme)

    if plot:
        # get feature importances
        band_gap_importances_and_names = list(sorted(zip(feature_strings,
                                                         band_gap_model.feature_importances_),
                                                     key=lambda x: x[1]))
        dielectric_constant_importances_and_names = list(sorted(zip(feature_strings,
                                                                    dielectric_constant_model.feature_importances_),
                                                                key=lambda x: x[1]))

        # Plot predicted vs actual for band gap
        plt.figure()
        plt.errorbar(band_gap_test, band_gap_pred, yerr=np.sqrt(band_gap_var), fmt='o', color='k', lw=0.5, zorder=0)
        p1 = plt.scatter(band_gap_test, band_gap_pred, color='b', zorder=10)
        ylim = plt.gca().get_ylim()
        xlim = plt.gca().get_xlim()
        p2, = plt.plot([ylim[0], ylim[1]], [ylim[0], ylim[1]], color='g')
        plt.legend([p1, p2], ["Actual", "Ideal"])
        plt.xlabel('Actual Band Gap (eV)', fontsize=16)
        plt.ylabel('Predicted Band Gap (eV)', fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.show()

        # Plot feature importances for band gap
        plt.figure()
        plt.barh(np.arange(10), [vals[1] for vals in band_gap_importances_and_names[:10]])
        plt.yticks(ticks=np.arange(10),
                   labels=[vals[0] for vals in band_gap_importances_and_names[:10]],
                   fontsize=11)
        plt.xlabel("Importance", fontsize=14)
        plt.show()

        # Plot distribution of model error for band gap
        pdf_x = np.linspace(-3, 3, 100)
        plt.figure()
        plt.plot(x, stats.norm.pdf(pdf_x, 0, 1))
        plt.hist((band_gap_pred - band_gap_test) / band_gap_std, density=True)
        plt.legend(["Ideal", "Actual"])
        plt.xlabel("Error / Uncertainty", fontsize=16)
        plt.ylabel("Probablity Density", fontsize=16)
        ylim = plt.gca().get_ylim()
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.show()

        # Plot predicted vs actual for dielectric constant
        plt.figure()
        plt.errorbar(dielectric_constant_test, dielectric_constant_pred, yerr=np.sqrt(dielectric_constant_var), fmt='o', color='k', lw=0.5, zorder=0)
        p1 = plt.scatter(dielectric_constant_test, dielectric_constant_pred, color='b', zorder=10)
        ylim = plt.gca().get_ylim()
        xlim = plt.gca().get_xlim()
        p2, = plt.plot([ylim[0], ylim[1]], [ylim[0], ylim[1]], color='g')
        plt.legend([p1, p2], ["Actual", "Ideal"])
        plt.xlabel('Actual Dielectric Constant', fontsize=16)
        plt.ylabel('Predicted Dielectric Constant', fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.show()

        # Plot feature importances for dielectric constant
        plt.figure()
        plt.barh(np.arange(10), [vals[1] for vals in dielectric_constant_importances_and_names[:10]])
        plt.yticks(ticks=np.arange(10),
                   labels=[vals[0] for vals in dielectric_constant_importances_and_names[:10]],
                   fontsize=11)
        plt.xlabel("Importance", fontsize=14)
        plt.show()

        # Plot distribution of model error for dielectric constant
        plt.figure()
        plt.plot(x, stats.norm.pdf(x, 0, 1))
        plt.hist((dielectric_constant_pred - dielectric_constant_test) / dielectric_constant_std, density=True)
        plt.legend(["Ideal", "Actual"])
        plt.xlabel("Error / Uncertainty", fontsize=16)
        plt.ylabel("Probablity Density", fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.ylim(ylim)
        plt.show()

    return band_gap, dielectric_constant # return the models and ndme scores

if __name__ == "__main__":
    band_gap, dielectric_constant = get_models(plot=True)
