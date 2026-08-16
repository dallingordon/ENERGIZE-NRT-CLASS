import numpy as np
import matplotlib.pyplot as plt
from ase.build import bulk
from ase.atoms import Atoms
from ase.visualize import view


def create_alcu_fcc(n_atoms=256, supercell=(4, 4, 4), perturbation=0.0, random_seed=None):
    """
    Create an AlCu FCC structure with approximately n_atoms by replicating the Al FCC unit cell
    and replacing half the atoms with Cu in a checkerboard pattern (simple mixing),
    then perturb positions with random displacements.
    """
    # Create FCC Al bulk unit cell
    al = bulk('Al', 'fcc', a=4.05)
    # replicate to supercell
    al_super = al.repeat(supercell)  # 4x4x4 means 4*4*4=64 unit cells * 4 atoms/unit cell = 256 atoms

    # Make a copy of the positions and chemical symbols list
    symbols = al_super.get_chemical_symbols()
    positions = al_super.get_positions()
    n = len(symbols)

    # Create a simple Al-Cu mixture:
    # Replace approx half of the atoms with Cu
    # We'll replace all atoms with even index with Al, odd with Cu for simplicity
    new_symbols = []
    for i in range(n):
        if i % 2 == 0:
            new_symbols.append('Al')
        else:
            new_symbols.append('Cu')

    alcu = Atoms(symbols=new_symbols, positions=positions, cell=al_super.cell, pbc=True)

    # Perturb positions if requested
    if perturbation > 0.0:
        np.random.seed(random_seed)
        displacement = (np.random.rand(len(alcu), 3) - 0.5) * 2 * perturbation  # uniform in [-perturb,perturb]
        alcu.set_positions(alcu.get_positions() + displacement)

    return alcu


def minimum_image_distance(vec, cell):
    """
    Apply minimum image convention distance calculation for vector vec in box defined by cell.
    Assumes orthogonal cell here for simplicity.
    """
    # For orthogonal cell:
    # Apply periodic boundary conditions using minimum image convention
    fractional = np.linalg.solve(cell.T, vec.T).T  # Convert to fractional coordinates
    fractional -= np.round(fractional)  # Wrap into [-0.5,0.5]
    cart = np.dot(fractional, cell)
    dist = np.linalg.norm(cart)
    return dist


def compute_rdf(atoms, r_max, dr, pairs=None):
    """
    Compute total RDF and pair-decomposed RDF for atoms.
    Parameters:
        atoms : ASE Atoms object with periodic boundary condition
        r_max : max radius to consider
        dr : bin width
        pairs : list of tuples (sym1, sym2); if None, compute all unique pairs, including total
    Returns:
        r: bin centers
        rdf_total: total radial distribution function
        rdf_pairs: dict with key (sym1,sym2) and RDF array as value
    """


    '''
    
    TO-DO
    
    '''

    return r, rdf_total, rdf_pairs


def plot_rdfs(r, rdf_total, rdf_pairs):
    import matplotlib.pyplot as plt

    # Plot total RDF
    plt.figure(figsize=(7, 5))
    plt.plot(r, rdf_total, label='Total RDF')
    plt.xlabel('Distance (Å)')
    plt.ylabel('g(r)')
    plt.title('Total Radial Distribution Function')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot pair RDFs each in its own subplot
    n_pairs = len(rdf_pairs)
    ncols = 2
    nrows = int(np.ceil(n_pairs / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False)
    fig.suptitle('Pair-Decomposed Radial Distribution Functions')

    for idx, ((s1, s2), rdf) in enumerate(rdf_pairs.items()):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row][col]
        ax.plot(r, rdf)
        ax.set_xlabel('Distance (Å)')
        ax.set_ylabel('g(r)')
        ax.set_title(f'{s1}-{s2} RDF')
        ax.grid(True)

    # Remove empty subplots
    for idx in range(n_pairs, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        fig.delaxes(axes[row][col])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def main():
    # Parameters
    n_atoms = 256  # approximately, from 4x4x4 fcc unit cells = 256 atoms exactly
    supercell = (4, 4, 4)  # 4x4x4 fcc cells (4 atoms per fcc cell)
    perturbation = 0.01  # max random displacement in Angstrom
    r_max = 10.0  # max radius for RDF calculation (angstrom)
    dr = 0.1  # bin width for RDF (angstrom)
    random_seed = 42

    atoms = create_alcu_fcc(n_atoms=n_atoms, supercell=supercell, perturbation=perturbation, random_seed=random_seed)
    print(f"Created AlCu FCC structure with {len(atoms)} atoms. Symbols count:")
    from collections import Counter
    print(Counter(atoms.get_chemical_symbols()))

    # Compute RDFs
    r, rdf_total, rdf_pairs = compute_rdf(atoms, r_max, dr)

    # Plot results
    plot_rdfs(r, rdf_total, rdf_pairs)


if __name__ == "__main__":
    main()