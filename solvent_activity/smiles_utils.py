from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np


def smiles_to_xyz(smiles):
    """
    Parse XYZ string and extract elements and coordinates.

    Args:
        xyz_string (str): XYZ string containing atom coordinates.

    Returns:
        elements (list): List of atom labels.
        coordinates (np.ndarray): Array of atom coordinates (shape: [num_atoms, 3]).
    """
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)  # Add hydrogens
    AllChem.EmbedMolecule(mol)  # Generate 3D coordinates
    AllChem.MMFFOptimizeMolecule(mol)  # Optimize geometry
    xyz_string = Chem.MolToXYZBlock(mol)  # Convert to XYZ format

    # Split XYZ string into lines, remove first 2
    data_list = xyz_string.strip().split("\n")[2:]

    # Extract elements and coordinates from each line
    elements = []
    coordinates = []

    for line in data_list:
        parts = line.split()
        elements.append(parts[0])  # Atom label
        coordinates.append(
            [float(parts[i]) for i in range(1, 4)]
        )  # X, Y, Z coordinates

    # Convert coordinates to NumPy array
    coordinates = np.array(coordinates)

    return elements, coordinates


