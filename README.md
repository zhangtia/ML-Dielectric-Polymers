# MSE482 - Citrine
## _Machine Informatics to Predict Polymer Properties_


## Quickstart

Machine learning model requires [Python Anaconda](https://www.anaconda.com/products/individual) to run.

If you are starting over, delete the previous virtual envrionment.

```sh
conda env remove -n MSE482
```

For first time initialization, create a virtual environment.
Ignore this step if you have created one already.

```sh
conda env create -f environment.yml
```

Activate the virtual environment and run the model.

```sh
conda activate MSE482
python3 core_model.py
```

Don't forget to exit out of the environment.

```sh
conda deactivate
```

## Overview
This repository contains the code and data required to reproduce our results in MSE482, where we use a Random Forest model to predict the band gap and dielectric constant of polymer repeating units with the aim of discovery novel polymeric dielectric materials for use in flexible, wearable electronics.

We use a dataset of 284 DFT-calculated band gaps and dielectric constants of polymeric repeating units as found in:

The original dataset can be found in ```4-block-polymer-data.csv```. The same dataset with SMILES strings added as a column can be found in ```4-block-polymer-data_SMILES.csv```. The code used to generate the SMILES strings can be found in ```get_monomer_SMILES.py```. We then featurize these repeating units using RDKit in ```featurize_monomers.py```.

The ```core_model.py``` file contains the main code used to build our model. ```sequential_learning.py``` contains a simulated sequential learning run of our model, where we test if our model can find promising candidates. We then predict the properties of new repeating units that are not found in our original dataset in ```get_new_monomers.py```, and select the best candidates in ```get_candidate_monomers.py```.
