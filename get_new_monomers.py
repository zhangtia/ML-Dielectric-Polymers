from itertools import permutations, combinations
import numpy as np

# This file contains functions to make the design space

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

