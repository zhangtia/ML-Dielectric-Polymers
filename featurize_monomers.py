import pandas as pd
import numpy as np
import rdkit.Chem.Descriptors
import rdkit.Chem.Descriptors3D
from rdkit.Chem import MolFromSmiles
import rdkit.Chem.AllChem
import os


def get_feature_strings():
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


def featurize_dataset(df):
    feature_strings = get_feature_strings()
    for feature, feature_name in zip(rdkit_features, feature_strings):
        def add_feature(row):
            # This creates a column in the dataset for each of the features
            if isinstance(row["SMILES"], str):
                # Make sure the SMILES string can be read
                if MolFromSmiles(row["SMILES"]) is not None:
                    # get rdkit molecule
                    mol = MolFromSmiles(row["SMILES"])
                    if feature_name in descriptor3D_strings:
                        # 3D features need the molecules conformation
                        try:
                            embed_id = rdkit.Chem.AllChem.EmbedMolecule(mol)
                            return float(feature(mol, confId=embed_id))
                        except:
                            molh = Chem.AddHs(mol) # some features need Hs for it to work
                            embed_id = rdkit.Chem.AllChem.EmbedMolecule(molh)
                            return float(feature(molh, confId=embed_id))
                    else:
                        return float(feature(mol))
            return None
        df[feature_name] = df.apply(add_feature, axis=1) # Create the column
    df.dropna(inplace=True) # Drop blank rows


def get_featurized_dataset(filename):
    if os.exists(filename):
        return pd.read_csv(filename)
    else:
        # # Read in the unfeaturized dataset
        df = pd.read_csv("4-block-polymer_data_SMILES.csv")
        featurize_dataset(df)
        df.to_csv(filename)
        return df


