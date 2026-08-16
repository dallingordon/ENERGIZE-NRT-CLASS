import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from dscribe.descriptors import SOAP, MBTR
from ase.io import read, write
from pymatgen.core import Structure
from mp_api.client import MPRester
from ase.constraints import FixAtoms
from ase import Atoms


# -------------------
# User settings
# -------------------
API_KEY = "YkfSuI6Em4JWu1DPx46brfuusaYVR06X"
CSV_PATH = r"C:/Users/jc112358/mp_metadata.csv"
STRUCT_DIR = "mp_structures"
FEATURES_FILE = r"C:/Users/jc112358/features.npz"

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
            atoms = Atoms(symbols=atoms.get_chemical_symbols(),
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
        atoms = Atoms(symbols=atoms.get_chemical_symbols(),
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
                #print(mid)
                atoms = get_structure_from_cif_or_api(mid, mpr)


                all_species = [
                    "H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar",
                    "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr"
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
                '''
                mbtr = MBTR(
                    species=all_species,
                    geometry={"function": "inverse_distance"},
                    grid={"min": 0, "max": 1, "n": 25, "sigma": 0.1},
                    weighting={"function": "exp", "scale": 0.5, "threshold": 1e-3},
                    periodic=True,
                    #normalization="l2",
                )
                '''
                if atoms is None:
                    continue

                feat = soap.create(atoms)  # (n_atoms, n_features)

                try:
                    feat = soap.create(atoms)  # (n_atoms, n_features)
                    #print(feat)
                    #feat = np.mean(feat, axis=0)  # average across atoms

                    X_list.append(feat)
                    y_list.append(energy)
                except Exception as e:
                    print(f"⚠️ SOAP failed for {mid}: {e}")
                    print(type(atoms))
                    continue

        X = np.vstack(X_list)
        y = np.array(y_list)

        print(f"✅ Features computed: {X.shape[0]} samples, {X.shape[1]} features")
        np.savez_compressed(features_file, X=X, y=y)
        return X, y


def train_gpr(X, y):
    """Train Gaussian Process Regression with PCA compression."""
    # PCA compression
    print("🔽 Applying PCA …")
    print(X)
    pca = PCA(n_components=min(N_COMPONENTS, X.shape[1]))
    X_reduced = X
    #pca.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_reduced, y, test_size=.9, random_state=42
    )

    # GPR kernel
    kernel = C(1.0, (1e-10, 1e3)) * RBF(length_scale=10) + WhiteKernel()
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=False)

    print("🚀 Training GPR …")
    gpr.fit(X_train, y_train)

    # Predictions
    y_train_pred, y_train_std = gpr.predict(X_train, return_std=True)
    y_test_pred, y_test_std = gpr.predict(X_test, return_std=True)

    # Metrics
    print("Train RMSE:", np.sqrt(mean_squared_error(y_train, y_train_pred)))
    print("Test  RMSE:", np.sqrt(mean_squared_error(y_test, y_test_pred)))
    print("Test  R^2 :", r2_score(y_test, y_test_pred))

    # Parity plots
    def parity_plot(y_true, y_pred, y_std, title, fname):
        plt.figure(figsize=(6, 6))
        plt.errorbar(y_true, y_pred, yerr=y_std, fmt="o", alpha=0.5)
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        plt.plot(lims, lims, "k--")
        plt.xlabel("DFT energy (eV)")
        plt.ylabel("Predicted energy (eV)")
        plt.title(title)
        plt.plot()

    parity_plot(y_train, y_train_pred, y_train_std, "Training Parity Plot", "train_parity.png")
    parity_plot(y_test, y_test_pred, y_test_std, "Test Parity Plot", "test_parity.png")

    return gpr


# -------------------
# Main
# -------------------
def main():
    os.makedirs(STRUCT_DIR, exist_ok=True)

    # Features
    X, y = compute_or_load_features(CSV_PATH, FEATURES_FILE)

    # Train GPR
    train_gpr(X, y)


if __name__ == "__main__":
    main()
