import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from dscribe.descriptors import SOAP
from ase.io import read
from mp_api.client import MPRester
from ase import Atoms

# -------------------
# User settings
# -------------------
API_KEY = "YOUR_KEY"
CSV_PATH = r"/path/to/mp_metadata.csv"
STRUCT_DIR = "mp_structures"
FEATURES_FILE = r"/path/to/features.npz"

# SOAP parameters
SOAP_NMAX = 2
SOAP_LMAX = 2
SOAP_RCUT = 4.0

# PCA
N_COMPONENTS = 10  # reduce dimensionality to avoid MemoryError


# -------------------
# Utilities
# -------------------
def get_structure_from_cif_or_api(material_id, mpr):
    """Load structure from local CIF if available, else fetch from MP API and save."""
    cif_path = os.path.join(STRUCT_DIR, f"{material_id}.cif")
    if os.path.exists(cif_path):
        try:
            atoms = read(cif_path)
            atoms = Atoms(
                symbols=atoms.get_chemical_symbols(),
                positions=atoms.get_positions(),
                cell=atoms.get_cell(),
                pbc=atoms.get_pbc()
            )
            return atoms
        except Exception as e:
            print(f"⚠️ Failed to read {cif_path}, refetching. Error: {e}")

    # Fallback: fetch from API
    try:
        struct = mpr.get_structure_by_material_id(material_id, conventional_unit_cell=True)
        struct.to(fmt="cif", filename=cif_path)  # save locally
        atoms = read(cif_path)
        atoms = Atoms(
            symbols=atoms.get_chemical_symbols(),
            positions=atoms.get_positions(),
            cell=atoms.get_cell(),
            pbc=atoms.get_pbc()
        )
        return atoms
    except Exception as e:
        print(f"❌ Could not fetch structure for {material_id}: {e}")
        return None


def compute_or_load_features(csv_path, features_file):
    """Compute SOAP features for all structures or load them from disk."""
    if os.path.exists(features_file):
        print(f"🔄 Loading precomputed features from {features_file}")
        data = np.load(features_file)
        return data["X"], data["y"]
    else:
        print("🧮 Computing SOAP features …")
        df = pd.read_csv(csv_path)

        X_list, y_list = [], []

        with MPRester(API_KEY) as mpr:
            for i, row in df.iterrows():
                mid, energy = row["material_id"], row["energy_per_atom"]
                atoms = get_structure_from_cif_or_api(mid, mpr)

                all_species = [
                    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
                    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br",
                    "Kr"
                ]

                soap = SOAP(
                    species=all_species,
                    r_cut=SOAP_RCUT,
                    n_max=SOAP_NMAX,
                    l_max=SOAP_LMAX,
                    sparse=False,
                    periodic=True,
                    average="inner",
                )

                if atoms is None:
                    continue

                try:
                    feat = soap.create(atoms)  # (n_atoms, n_features)
                    X_list.append(feat)
                    y_list.append(energy)
                except Exception as e:
                    print(f"⚠️ SOAP failed for {mid}: {e}")
                    continue

        X = np.vstack(X_list)
        y = np.array(y_list)

        print(f"✅ Features computed: {X.shape[0]} samples, {X.shape[1]} features")
        np.savez_compressed(features_file, X=X, y=y)
        return X, y


# -------------------
# Models
# -------------------
def train_gpr(X_train, X_test, y_train, y_test):
    """Train Gaussian Process Regression."""







    return y_train_pred, y_train_std, y_test_pred, y_test_std


def train_nn(X_train, X_test, y_train, y_test):
    """Train simple feedforward NN."""








    return y_train_pred, y_test_pred


# -------------------
# Visualization
# -------------------
def plot_results(y_train, y_train_pred_gpr, y_test, y_test_pred_gpr,
                 y_train_pred_nn, y_test_pred_nn):
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    def parity(ax, y_true, y_pred, title):
        ax.scatter(y_true, y_pred, alpha=0.5)
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, "k--")
        ax.set_xlabel("DFT energy (eV)")
        ax.set_ylabel("Predicted energy (eV)")
        ax.set_title(title)

    parity(axes[0, 0], y_train, y_train_pred_gpr, "GPR Training")
    parity(axes[0, 1], y_test, y_test_pred_gpr, "GPR Test")
    parity(axes[1, 0], y_train, y_train_pred_nn, "NN Training")
    parity(axes[1, 1], y_test, y_test_pred_nn, "NN Test")

    plt.tight_layout()
    plt.show()


# -------------------
# Main
# -------------------
def main():
    os.makedirs(STRUCT_DIR, exist_ok=True)

    # Features
    X, y = compute_or_load_features(CSV_PATH, FEATURES_FILE)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.8, random_state=1958275
    )

    # Train GPR
    y_train_pred_gpr, y_train_std, y_test_pred_gpr, y_test_std = train_gpr(X_train, X_test, y_train, y_test)

    # Train NN
    y_train_pred_nn, y_test_pred_nn = train_nn(X_train, X_test, y_train, y_test)

    # Plot results
    plot_results(y_train, y_train_pred_gpr, y_test, y_test_pred_gpr,
                 y_train_pred_nn, y_test_pred_nn)


if __name__ == "__main__":
    main()
