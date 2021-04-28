import pandas as pd
from itertools import combinations, permutations
import numpy as np
import ast
from sklearn.kernel_ridge import KernelRidge as KR

def make_block_str(ordered_blocks):
    # Turn a list of blocks into a single string
    return ''.join([block + '-' for block in ordered_blocks])[:-1]


def block_to_smiles(block):
    # Return the SMILES string that corresponds to each block
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

def get_smiles_string(row):
    # Get the SMILES string for a given monomer
    blocks = row["4-block polymer"]
    if isinstance(blocks, str):
        return ''.join([block_to_smiles(block) for block in blocks.split('-')])

df = pd.read_csv("4-block-polymer_data.csv")

# This creates SMILES strings to be uploaded to citrination
df["SMILES"] = df.apply(get_smiles_string, axis=1)
df.to_csv("4-block-polymer_data_SMILES.csv")