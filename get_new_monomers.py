import pandas as pd
import numpy as np
import forestci
from itertools import permutations, combinations
import featurize_monomers
import scipy.stats as stats
from core_model import get_models

def block_to_smiles(block):
    """ Convert the formula of a block to a SMILES string """
    if block == 'CH2':
        return 'C'
    elif block == 'NH':
        return 'N'
    elif block == 'O':
        return 'O'
    elif block == 'C6H4':
        return 'C1=CC=CC=C1'
    elif block == 'C4H2S':
        return 'C1=CSC=C1'
    elif block == 'CO':
        return 'C(=O)'
    elif block == 'CS':
        return 'C(=S)'


def get_smiles_string(ordered_blocks):
    """ Concatenate the SMILES strings of each block """
    return ''.join([block_to_smiles(block) for block in ordered_blocks])


def O_N_adjacent(molecule):
    # Determine if there is an O-N, N-O, O-O, or N-N bond
    if molecule[0] in ['O', 'NH'] and molecule[:-1] in ['O', 'NH']:
        return True
    for i in range(len(molecule) - 1):
        if molecule[i] in ['O', 'NH'] and molecule[i + 1] in ['O', 'NH']:
            return True
    return False


def get_all_permutations(blocks, n):
    # Get permutations where every element is unique
    unique_permutations = list(permutations(blocks, n))
    # Get pairs of number that add up to n
    n_pairs = []
    for i in range(1, int(np.floor(n / 2))):
        n_pairs.append((i, n - i))
    # Create sub-permutations
    for i, j in n_pairs:
        for i_sub_molecule in permutations(blocks, i):
            for j_sub_molecule in permutations(blocks, j):
                # Combine sub-permutations into n-block polymer
                combined_permutation = list(i_sub_molecule)
                combined_permutation.extend(list(j_sub_molecule))
                if tuple(combined_permutation) not in unique_permutations:
                    # Append newly found permutations
                    unique_permutations.append(tuple(combined_permutation))
    return unique_permutations


def get_new_SMILES(known_SMILES, blocks, n_permutations):
    # Find the SMILES strings we don't have in the dataset
    all_permutations = get_all_permutations(blocks, n_permutations)
    new_SMILES = []
    new_blocks = []
    for molecule in all_permutations:
        if not O_N_adjacent(molecule):
            SMILES_str = get_smiles_string(molecule)
            if SMILES_str not in known_SMILES:
                new_SMILES.append(SMILES_str)
                new_blocks.append('-'.join(molecule))
    return new_SMILES, new_blocks


feature_strings = featurize_monomers.get_feature_strings()
band_gap, dielectric_constant = get_models(plot=False)
band_gap_model = band_gap.model
dielectric_constant_model = dielectric_constant.model

known_SMILES = list(df["SMILES"]) # Get list of SMILES strings we already have
blocks = sorted(['CH2', 'C6H4', 'NH', 'CO', 'CS', 'C4H2S', 'O'])

new_df = pd.DataFrame() # Create a new dataset of our predicted values
SMILES_and_blocks = get_new_SMILES(known_SMILES, blocks, 4)
new_df["SMILES"] = SMILES_and_blocks[0]
new_df["4-block polymer"] = SMILES_and_blocks[1]
featurize_monomers.featurize_dataset(new_df) # Featurize the new monomers
new_feature_values = new_df[feature_strings].values # Get feature values

# Get predictions for the new monomers
predicted_bandgaps = band_gap_model.predict(new_feature_values)
bandgap_var = forestci.random_forest_error(band_gap_model, feature_values1_train, new_feature_values)
bandgap_std = np.sqrt(sum(bandgap_var) / len(bandgap_var)) # Get standard deviation

predicted_dielectric_constants = dielectric_constant_model.predict(new_feature_values)
dielectric_var = forestci.random_forest_error(dielectric_constant_model, feature_values2_train, new_feature_values)
dielectric_std = np.sqrt(sum(dielectric_var) / len(dielectric_var)) # Get standard deviation

new_df["Band Gap (eV)"] = predicted_bandgaps
new_df["Band Gap Uncertainty (eV)"] = np.sqrt(bandgap_var)
new_df["Total Dielectric Constant"] = predicted_dielectric_constants
new_df["Total Dielectric Constant Uncertainty"] = np.sqrt(dielectric_var)


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
new_df.to_csv("new_4-block_monomers.csv") # Save values
