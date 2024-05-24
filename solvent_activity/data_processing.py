import os
import pandas as pd
import numpy as np
import morfeus as mf
import traceback
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from solvent_activity.smiles_utils import smiles_to_xyz
from tqdm import tqdm
def morfeus_descriptors(string_list):
    all_descriptors = []
    for index, smiles in tqdm(enumerate(string_list)):
        elements, coordinates = smiles_to_xyz(smiles)

        # Dispersion
        disp = mf.Dispersion(elements, coordinates)
        disp_descriptors = [disp.area, disp.p_int, disp.volume]

        # Solvent accessible surface area
        sasa = mf.SASA(elements, coordinates)
        sasa_descriptors = [sasa.area, sasa.volume]

        # Electronic parameters
        xtb = mf.XTB(elements, coordinates)
        xtb_descriptors = [
            xtb.get_dipole()[0],
            xtb.get_dipole()[1],
            xtb.get_dipole()[2],
            xtb.get_ea(corrected=True),
            xtb.get_global_descriptor("electrophilicity", corrected=True),
            xtb.get_global_descriptor("nucleophilicity", corrected=True),
            xtb.get_global_descriptor("electrofugality", corrected=True),
            xtb.get_global_descriptor("nucleofugality", corrected=True),
            xtb.get_homo(),
            xtb.get_lumo(),
            xtb.get_ip(corrected=True),
        ]

        list_all = disp_descriptors + sasa_descriptors + xtb_descriptors
        all_descriptors.append(np.array(list_all))
    return np.array(all_descriptors)


def get_morfeus_descriptors(df):
    # Convert SMILES to morfeus descriptors
    # Step 1: Identify unique SMILES
    unique_smiles_solute = df['Solute_SMILES'].unique()
    unique_smiles_solvent = df['Solvent_SMILES'].unique()

    # Step 2: Calculate descriptors for unique SMILES
    unique_solute_descriptors =  {smiles: desc for smiles, desc in zip(unique_smiles_solute, morfeus_descriptors(unique_smiles_solute))}
    unique_solvent_descriptors =  {smiles: desc for smiles, desc in zip(unique_smiles_solvent, morfeus_descriptors(unique_smiles_solvent))}

    # Step 3: Map the results back to the original DataFrame
    solute_fps= df['Solute_SMILES'].map(unique_solute_descriptors).tolist()
    solvent_fps= df['Solvent_SMILES'].map(unique_solvent_descriptors).tolist()

    X = np.concatenate([np.array(solvent_fps), np.array(solute_fps)], axis=1)
    return X, solvent_fps, solute_fps


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


def split_csv(csv_file, output_dir):
    # Load the CSV file
    data = pd.read_csv(csv_file)
    conc_morfeus_descirptors, solvent_morfeus_descriptors, solute_morfeus_descriptors = get_morfeus_descriptors(data)


    conc_morgan_fps, solvent_morgan_fps, solute_morgan_fps = get_morgan_fingerprints(
        data
    )
    conc_torsion_fps, solvent_torsion_fps, solute_torsion_fps = (
        get_torsion_fingerprints(data)
    )
    conc_descirptors, solvent_descriptors, solute_descriptors = get_descriptors(data)


    data["conc_morgan_fp"] = conc_morgan_fps.tolist()
    data["conc_torsion_fp"] = conc_torsion_fps.tolist()
    data["conc_descirptors"] = conc_descirptors.tolist()
    data["conc_morfeus_descirptors"] = conc_morfeus_descirptors.tolist()

    # Save solvent fingerprints and descriptors
    data["solvent_morgan_fp"] = solvent_morgan_fps
    data["solvent_torsion_fp"] = solvent_torsion_fps
    data["solvent_descriptors"] = solvent_descriptors
    data["solvent_morfeus_descriptors"] = solvent_morfeus_descriptors

    # Save solute fingerprints and descriptors
    data["solute_morgan_fp"] = solute_morgan_fps
    data["solute_torsion_fp"] = solute_torsion_fps
    data["solute_descriptors"] = solute_descriptors
    data["solute_morfeus_descriptors"] = solute_morfeus_descriptors

    # Combine solvent and solute SMILES 
    data["solvent_solute_smiles"] = data["Solvent_SMILES"] + '.' + data["Solute_SMILES"]


    # Split the data into train and test sets (80-20 split)
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # Save the test set
    test_data.to_csv(
        os.path.join(output_dir, f"test_{os.path.basename(csv_file)}"), index=False
    )

    # Perform 5-fold cross-validation on the training set
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for i, (train_idx, val_idx) in enumerate(kf.split(train_data)):
        train_fold = train_data.iloc[train_idx]
        val_fold = train_data.iloc[val_idx]

        y_train = train_fold["Literature"]
        y_val = val_fold["Literature"]

        y_train_minmax, min_max_scaler = scale_target(y_train, "minmax")
        y_train_standard, standard_scaler = scale_target(y_train, "standard")

        y_val_minmax = min_max_scaler.transform(y_val.values.reshape(-1, 1))
        y_val_standard = standard_scaler.transform(y_val.values.reshape(-1, 1))

        val_fold["y_minmax"] = y_val_minmax
        val_fold["y_standard"] = y_val_standard

        train_fold["y_minmax"] = y_train_minmax
        train_fold["y_standard"] = y_train_standard

        # Create the output directory for this fold
        fold_dir = os.path.join(output_dir, f"split_{i+1}")
        os.makedirs(fold_dir, exist_ok=True)

        # Save the train and validation sets for this fold
        train_fold.to_parquet(
            os.path.join(fold_dir, "train_database_IAC_ln_clean.parquet"), index=False
        )
        val_fold.to_parquet(
            os.path.join(fold_dir, "val_database_IAC_ln_clean.parquet"), index=False
        )

        pd.to_pickle(
            min_max_scaler,
            os.path.join(fold_dir, "min_max_scaler.pkl"),
        )

        pd.to_pickle(
            standard_scaler,
            os.path.join(fold_dir, "standard_scaler.pkl"),
        )


if __name__ == "__main__":
    # Example usage
    csv_file = "./data/raw/database_IAC_ln_clean.csv"
    output_dir = "./data/splits/"

    split_csv(csv_file, output_dir)
