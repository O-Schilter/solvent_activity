import os
import traceback

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

from sklearn.preprocessing import MinMaxScaler, StandardScaler


def getMolDescriptors(mol, missingVal=None):
    """calculate the full list of descriptors for a molecule

    missingVal is used if the descriptor cannot be calculated
    """
    res = []
    for nm, fn in Descriptors._descList:
        # some of the descriptor fucntions can throw errors if they fail, catch those here:
        try:
            val = fn(mol)
        except:
            # print the error message:
            traceback.print_exc()
            # and set the descriptor value to whatever missingVal is
            val = missingVal
        res.append(val)
    return res


def get_morgan_fingerprints(df, fp_size=1024):
    # Convert SMILES to Morgan fingerprints
    solvent_fps = [
        np.array(
            AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smile), fp_size)
        )
        for smile in df["Solvent_SMILES"]
    ]
    solute_fps = [
        np.array(
            AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smile), fp_size)
        )
        for smile in df["Solute_SMILES"]
    ]

    # # Convert fingerprints to NumPy arrays
    # solvent_fps = np.array(solvent_fps)
    # solute_fps = np.array(solute_fps)

    # Concatenate solvent and solute fingerprints
    X = np.concatenate([solvent_fps, solute_fps], axis=1)
    return X, solvent_fps, solute_fps


def get_torsion_fingerprints(df):
    solvent_fps = [
        np.array(
            AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect(
                Chem.MolFromSmiles(smile)
            )
        )
        for smile in df["Solvent_SMILES"]
    ]
    solute_fps = [
        np.array(
            AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect(
                Chem.MolFromSmiles(smile)
            )
        )
        for smile in df["Solute_SMILES"]
    ]
    # solvent_fps = np.array(solvent_fps)
    # solute_fps = np.array(solute_fps)
    # Concatenate solvent and solute fingerprints
    X = np.concatenate([solvent_fps, solute_fps], axis=1)
    return X, solvent_fps, solute_fps


def get_descriptors(df):
    solvent_descriptors = [
        list(getMolDescriptors(Chem.MolFromSmiles(smile)))
        for smile in df["Solvent_SMILES"]
    ]
    solute_descriptors = [
        list(getMolDescriptors(Chem.MolFromSmiles(smile)))
        for smile in df["Solute_SMILES"]
    ]
    # solvent_descriptors = np.array(solvent_descriptors)
    # solute_descriptors = np.array(solute_descriptors)
    # Concatenate solvent and solute fingerprints
    X = np.concatenate([solvent_descriptors, solute_descriptors], axis=1)
    return X, solvent_descriptors, solute_descriptors


def scale_target(y, scaler_type):
    if scaler_type == "minmax":
        scaler = MinMaxScaler()
    elif scaler_type == "standard":
        scaler = StandardScaler()
    else:
        return y

    y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))
    return y_scaled.ravel(), scaler


def main():
    splits_dir = "./data/splits"
    output_dir = "./data/processed"

    for split_dir in os.listdir(splits_dir):
        split_path = os.path.join(splits_dir, split_dir)
        if os.path.isdir(split_path):
            processed_data = []
            print(os.listdir(split_path))
            for filename in reversed(os.listdir(split_path)):
                if filename.startswith("train_") or filename.startswith("val_"):
                    output_split_dir = os.path.join(output_dir, split_dir)
                    os.makedirs(output_split_dir, exist_ok=True)
                    print("filename", filename)
                    csv_file = os.path.join(split_path, filename)
                    df = pd.read_csv(csv_file)

                    # Add the target variable
                    y = df["Literature"]
                    y_original = y.copy()
                    if filename.startswith("train_"):
                        y_minmax, min_max_scaler = scale_target(y, "minmax")
                        y_standard, standard_scaler = scale_target(y, "standard")
                    else:
                        min_max_scaler = pd.read_pickle(
                            os.path.join(output_split_dir, "min_max_scaler.pkl")
                        )
                        y_minmax = min_max_scaler.transform(y.values.reshape(-1, 1))
                        standard_scaler = pd.read_pickle(
                            os.path.join(output_split_dir, "standard_scaler.pkl")
                        )
                        y_standard = standard_scaler.transform(y.values.reshape(-1, 1))

                    conc_morgan_fps, solvent_morgan_fps, solute_morgan_fps = (
                        get_morgan_fingerprints(df)
                    )
                    conc_torsion_fps, solvent_torsion_fps, solute_torsion_fps = (
                        get_torsion_fingerprints(df)
                    )
                    conc_descirptors, solvent_descriptors, solute_descriptors = (
                        get_descriptors(df)
                    )

                    data = pd.DataFrame()

                    # Save concatenated fingerprints and descriptors
                    data["conc_morgan_fp"] = conc_morgan_fps.tolist()
                    data["conc_torsion_fp"] = conc_torsion_fps.tolist()
                    data["conc_descirptors"] = conc_descirptors.tolist()

                    # Save solvent fingerprints and descriptors
                    data["solvent_morgan_fp"] = solvent_morgan_fps
                    data["solvent_torsion_fp"] = solvent_torsion_fps
                    data["solvent_descriptors"] = solvent_descriptors

                    # Save solute fingerprints and descriptors
                    data["solute_morgan_fp"] = solute_morgan_fps
                    data["solute_torsion_fp"] = solute_torsion_fps
                    data["solute_descriptors"] = solute_descriptors
                    data["y_original"] = y_original
                    data["y_minmax"] = y_minmax
                    data["y_standard"] = y_standard

                    output_file = os.path.join(output_split_dir, f"{filename}.parquet")
                    data.to_parquet(output_file)

                    pd.to_pickle(
                        min_max_scaler,
                        os.path.join(output_split_dir, "min_max_scaler.pkl"),
                    )

                    pd.to_pickle(
                        standard_scaler,
                        os.path.join(output_split_dir, "standard_scaler.pkl"),
                    )


if __name__ == "__main__":
    main()
