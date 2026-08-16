# requirements: pip install mp_api pymatgen

import os
import csv
from tqdm import tqdm
from itertools import combinations
import requests
from mp_api.client import MPRester
# ----------------------------
# CONFIG
# ----------------------------
API_KEY = "YkfSuI6Em4JWu1DPx46brfuusaYVR06X"   # <-- insert your Materials Project API key
OUT_DIR = "mp_structures"
CSV_PATH = "mp_metadata.csv"
CHUNK_SIZE = 1000  # number of results per API request

# Elements to consider (you can extend this)
ELEMENTS = [
    "H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar",
    "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr"
]

# Max number of elements per chemical system to query (1=unary, 2=binary, etc.)
MAX_ELEMENTS = 2

# ----------------------------
# Fetch material IDs by chemical system
# ----------------------------
def fetch_all_material_ids(mpr):
    all_ids = set()

    for n in range(1, MAX_ELEMENTS+1):
        for combo in tqdm(combinations(ELEMENTS, n), desc=f"Fetching {n}-element systems",disable=True):
            try:
                results = mpr.materials.summary.search(
                    chemsys="-".join(combo),
                    fields=["material_id"],
                    chunk_size=CHUNK_SIZE
                )
                for doc in results:
                    all_ids.add(doc.material_id)
            except Exception as e:
                print(f"⚠️ Failed for {combo}: {e}")

    return list(all_ids)

# ----------------------------
# Fetch structures and energies
# ----------------------------
def fetch_and_save_data(material_ids, mpr):
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["material_id", "formula_pretty", "energy_per_atom", "cif_path"])

        for mid in tqdm(material_ids, desc="Downloading structures",disable=True):
            try:
                summary = mpr.materials.summary.search(
                    material_ids=[mid],
                    fields=["material_id", "formula_pretty", "energy_per_atom"]
                )
                if not summary:
                    continue
                doc = summary[0]

                struct = mpr.get_structure_by_material_id(mid)
                cif_path = os.path.join(OUT_DIR, f"{mid}.cif")
                struct.to(fmt="cif", filename=cif_path)

                writer.writerow([mid, doc.formula_pretty, doc.energy_per_atom, cif_path])

            except Exception as e:
                print(f"⚠️ Failed for {mid}: {e}")

# ----------------------------
# Main
# ----------------------------
def main():
    with MPRester(API_KEY) as mpr:
        print("Fetching all material IDs …")
        material_ids = fetch_all_material_ids(mpr)
        print(f"✅ Total materials found: {len(material_ids)}")

        print("Fetching structures and energies …")
        fetch_and_save_data(material_ids, mpr)
        print(f"✅ Done! Structures → {OUT_DIR}, metadata → {CSV_PATH}")

if __name__ == "__main__":
    main()
