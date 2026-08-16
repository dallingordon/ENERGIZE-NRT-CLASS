"""


Purpose
-------
This example introduces a graph neural network (GNN) using PyTorch.

The goal is to predict the boiling temperature of small synthetic organic
molecules.

Unlike the earlier regression example, each input is not represented by one
fixed feature vector. Instead, each molecule is represented as a GRAPH:

    atoms  -> nodes
    bonds  -> edges

The graph neural network learns from the local atomic environments and then
combines information from all atoms to make one molecular prediction.

This script performs the following steps:

    1. Generate a synthetic molecular dataset.
    2. Build a graph for every molecule.
    3. Create node features for the atoms.
    4. Create an adjacency matrix from the molecular bonds.
    5. Split the molecules into training and test sets.
    6. Build a simple GNN directly in PyTorch.
    7. Train the GNN to predict boiling temperature.
    8. Save training diagnostics.
    9. Create separate training and test parity plots.

IMPORTANT
---------
The molecules and boiling temperatures in this example are SYNTHETIC.

The values are generated to follow physically reasonable trends for small
organic molecules, but they are not intended to reproduce an experimental
boiling-point database.

This example intentionally does NOT use PyTorch Geometric.

The graph construction and message passing are written directly in the script
so that the individual steps are visible.
"""

from pathlib import Path
import math
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# ===========================================================================
# 1. PATHS AND SETTINGS
# ===========================================================================

SCRIPT_DIRECTORY = Path(__file__).resolve().parent

RESULTS_DIRECTORY = (
    SCRIPT_DIRECTORY
    / "gnn_results"
)

RESULTS_DIRECTORY.mkdir(
    exist_ok=True
)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

RANDOM_SEED = 42

random.seed(
    RANDOM_SEED
)

np.random.seed(
    RANDOM_SEED
)

torch.manual_seed(
    RANDOM_SEED
)


# ---------------------------------------------------------------------------
# Dataset and training settings
# ---------------------------------------------------------------------------

NUMBER_OF_MOLECULES = 300

TEST_FRACTION = 0.20

NUMBER_OF_EPOCHS = 220

LEARNING_RATE = 0.003

PRINT_EVERY = 20


# This introductory example intentionally uses the CPU.

DEVICE = torch.device(
    "cpu"
)


print("=" * 75)
print("PyTorch Graph Neural Network: Molecular Boiling Temperature")
print("=" * 75)
print()

print("Device:")
print(DEVICE)
print()

print("Results directory:")
print(RESULTS_DIRECTORY)
print()


# ===========================================================================
# 2. ATOM INFORMATION
# ===========================================================================

# This introductory dataset uses only four atom types.
#
# Hydrogen atoms are NOT represented as graph nodes.
# They are treated as implicit atoms, which keeps the graphs small.

ATOM_TYPES = [
    "C",
    "O",
    "N",
    "Cl",
]


# Approximate atomic masses are included as one node feature.

ATOMIC_MASS = {
    "C": 12.01,
    "O": 16.00,
    "N": 14.01,
    "Cl": 35.45,
}


# ===========================================================================
# 3. MOLECULE FAMILIES
# ===========================================================================

# Each synthetic molecule belongs to one recognizable organic family.
#
# These families were chosen because their boiling temperatures follow
# different broad chemical trends.

MOLECULE_FAMILIES = [
    "alkane",
    "alcohol",
    "amine",
    "ketone",
    "nitrile",
    "chloroalkane",
]


# ===========================================================================
# 4. HELPER FUNCTION: MOLECULAR FORMULA
# ===========================================================================

def create_formula(
    number_of_carbons,
    family
):
    """
    Create a simple molecular formula for the synthetic molecule.

    Hydrogen atoms are implicit in the graph, but including the formula makes
    the generated dataset easier to read.
    """

    n = number_of_carbons


    if family == "alkane":

        number_of_hydrogens = 2 * n + 2

        formula = (
            f"C{n}"
            f"H{number_of_hydrogens}"
        )


    elif family == "alcohol":

        number_of_hydrogens = 2 * n + 2

        formula = (
            f"C{n}"
            f"H{number_of_hydrogens}"
            "O"
        )


    elif family == "amine":

        number_of_hydrogens = 2 * n + 3

        formula = (
            f"C{n}"
            f"H{number_of_hydrogens}"
            "N"
        )


    elif family == "ketone":

        number_of_hydrogens = 2 * n

        formula = (
            f"C{n}"
            f"H{number_of_hydrogens}"
            "O"
        )


    elif family == "nitrile":

        number_of_hydrogens = 2 * n - 1

        formula = (
            f"C{n}"
            f"H{number_of_hydrogens}"
            "N"
        )


    else:

        # chloroalkane

        number_of_hydrogens = 2 * n + 1

        formula = (
            f"C{n}"
            f"H{number_of_hydrogens}"
            "Cl"
        )


    return formula


# ===========================================================================
# 5. HELPER FUNCTION: SYNTHETIC BOILING TEMPERATURE
# ===========================================================================

def synthetic_boiling_temperature(
    number_of_carbons,
    family,
    is_branched
):
    """
    Generate a physically reasonable synthetic boiling temperature.

    The equation is NOT a real chemistry model.

    It simply reproduces several important qualitative trends:

        1. Larger molecules generally boil at higher temperatures.
        2. Polar functional groups generally increase boiling temperature.
        3. Hydrogen-bonding groups such as alcohols increase boiling
           temperature strongly.
        4. Branching tends to lower boiling temperature somewhat.
        5. Random noise prevents the problem from being perfectly
           deterministic.
    """

    n = number_of_carbons


    # -----------------------------------------------------------------------
    # Base hydrocarbon trend
    # -----------------------------------------------------------------------

    # This produces an increasing but slightly nonlinear trend with carbon
    # count.

    base_temperature = (
        93.0
        + 55.0 * n
        - 2.0 * n ** 2
    )


    # -----------------------------------------------------------------------
    # Functional-group contribution
    # -----------------------------------------------------------------------

    if family == "alkane":

        functional_group_effect = 0.0


    elif family == "alcohol":

        # Alcohols have strong intermolecular hydrogen bonding.

        functional_group_effect = (
            180.0
            - 12.0 * n
        )


    elif family == "amine":

        functional_group_effect = (
            120.0
            - 6.0 * n
        )


    elif family == "ketone":

        functional_group_effect = (
            120.0
            - 8.0 * n
        )


    elif family == "nitrile":

        functional_group_effect = (
            190.0
            - 15.0 * n
        )


    else:

        # Chloroalkanes are heavier and more polarizable than the
        # corresponding hydrocarbons.

        functional_group_effect = (
            115.0
            - 8.0 * n
        )


    # -----------------------------------------------------------------------
    # Branching contribution
    # -----------------------------------------------------------------------

    if is_branched:

        branching_effect = -15.0

    else:

        branching_effect = 0.0


    # -----------------------------------------------------------------------
    # Random variation
    # -----------------------------------------------------------------------

    random_variation = np.random.normal(
        loc=0.0,
        scale=10.0
    )


    boiling_temperature = (
        base_temperature
        + functional_group_effect
        + branching_effect
        + random_variation
    )


    # Keep the synthetic values in a reasonable teaching range.

    boiling_temperature = np.clip(
        boiling_temperature,
        170.0,
        620.0
    )


    return float(
        boiling_temperature
    )


# ===========================================================================
# 6. BUILD A CARBON SKELETON
# ===========================================================================

def build_carbon_skeleton(
    number_of_carbons,
    is_branched
):
    """
    Build the carbon portion of the molecular graph.

    Returns
    -------
    atom_types
        List containing one atom symbol for every graph node.

    bonds
        List of tuples:
            (node_1, node_2, bond_order)

    positions
        Simple 2D positions used only for drawing the example graph.

    Example
    -------

    A four-carbon linear skeleton:

        C - C - C - C

    A branched four-carbon skeleton:

            C
            |
        C - C - C
    """

    atom_types = []

    bonds = []

    positions = []


    # -----------------------------------------------------------------------
    # Linear molecule
    # -----------------------------------------------------------------------

    if (
        not is_branched
        or number_of_carbons < 4
    ):

        for carbon_index in range(
            number_of_carbons
        ):

            atom_types.append(
                "C"
            )

            positions.append(
                (
                    float(carbon_index),
                    0.0
                )
            )


        for carbon_index in range(
            number_of_carbons - 1
        ):

            bonds.append(
                (
                    carbon_index,
                    carbon_index + 1,
                    1
                )
            )


    # -----------------------------------------------------------------------
    # Molecule with one carbon branch
    # -----------------------------------------------------------------------

    else:

        main_chain_length = (
            number_of_carbons
            - 1
        )


        # Build the main carbon chain.

        for carbon_index in range(
            main_chain_length
        ):

            atom_types.append(
                "C"
            )

            positions.append(
                (
                    float(carbon_index),
                    0.0
                )
            )


        for carbon_index in range(
            main_chain_length - 1
        ):

            bonds.append(
                (
                    carbon_index,
                    carbon_index + 1,
                    1
                )
            )


        # Add one branch carbon.

        branch_node_index = len(
            atom_types
        )

        atom_types.append(
            "C"
        )


        # Attach the branch somewhere away from the ends of the main chain.

        possible_branch_positions = list(
            range(
                1,
                main_chain_length - 1
            )
        )


        branch_attachment_index = random.choice(
            possible_branch_positions
        )


        bonds.append(
            (
                branch_attachment_index,
                branch_node_index,
                1
            )
        )


        branch_x = float(
            branch_attachment_index
        )

        positions.append(
            (
                branch_x,
                1.0
            )
        )


    return (
        atom_types,
        bonds,
        positions
    )


# ===========================================================================
# 7. ADD THE FUNCTIONAL GROUP
# ===========================================================================

def add_functional_group(
    atom_types,
    bonds,
    positions,
    family
):
    """
    Add one simple functional group to the carbon skeleton.

    The graph contains explicit C, O, N, and Cl atoms.

    Hydrogen atoms remain implicit.
    """

    number_of_carbon_nodes = sum(
        atom == "C"
        for atom in atom_types
    )


    # Carbon nodes are created before heteroatoms, so the first and last carbon
    # indices are easy to identify.

    first_carbon = 0

    last_carbon = (
        number_of_carbon_nodes
        - 1
    )


    if family == "alkane":

        # Nothing needs to be added.

        return


    elif family == "alcohol":

        new_node = len(
            atom_types
        )

        atom_types.append(
            "O"
        )

        bonds.append(
            (
                last_carbon,
                new_node,
                1
            )
        )

        last_position = positions[
            last_carbon
        ]

        positions.append(
            (
                last_position[0] + 0.7,
                last_position[1] + 0.8
            )
        )


    elif family == "amine":

        new_node = len(
            atom_types
        )

        atom_types.append(
            "N"
        )

        bonds.append(
            (
                last_carbon,
                new_node,
                1
            )
        )

        last_position = positions[
            last_carbon
        ]

        positions.append(
            (
                last_position[0] + 0.7,
                last_position[1] + 0.8
            )
        )


    elif family == "ketone":

        new_node = len(
            atom_types
        )

        atom_types.append(
            "O"
        )


        # Place the carbonyl oxygen near the middle carbon.

        carbonyl_carbon = int(
            number_of_carbon_nodes / 2
        )


        bonds.append(
            (
                carbonyl_carbon,
                new_node,
                2
            )
        )


        carbon_position = positions[
            carbonyl_carbon
        ]

        positions.append(
            (
                carbon_position[0],
                carbon_position[1] + 1.0
            )
        )


    elif family == "nitrile":

        new_node = len(
            atom_types
        )

        atom_types.append(
            "N"
        )

        bonds.append(
            (
                last_carbon,
                new_node,
                3
            )
        )

        last_position = positions[
            last_carbon
        ]

        positions.append(
            (
                last_position[0] + 1.0,
                last_position[1]
            )
        )


    else:

        # chloroalkane

        new_node = len(
            atom_types
        )

        atom_types.append(
            "Cl"
        )

        bonds.append(
            (
                first_carbon,
                new_node,
                1
            )
        )

        first_position = positions[
            first_carbon
        ]

        positions.append(
            (
                first_position[0] - 1.0,
                first_position[1]
            )
        )


# ===========================================================================
# 8. CREATE AN ADJACENCY MATRIX
# ===========================================================================

def create_adjacency_matrix(
    number_of_nodes,
    bonds
):
    """
    Convert the bond list into an adjacency matrix.

    adjacency[i, j] = 1 means that atom i and atom j are bonded.

    The molecular graph is undirected, so:

        adjacency[i, j] = adjacency[j, i]
    """

    adjacency = np.zeros(
        (
            number_of_nodes,
            number_of_nodes
        ),
        dtype=np.float32
    )


    for node_1, node_2, bond_order in bonds:

        adjacency[
            node_1,
            node_2
        ] = 1.0

        adjacency[
            node_2,
            node_1
        ] = 1.0


    return adjacency


# ===========================================================================
# 9. CREATE NODE FEATURES
# ===========================================================================

def create_node_features(
    atom_types,
    bonds
):
    """
    Create one feature vector for every atom.

    Node features
    -------------

    Features 1-4:
        One-hot atom identity:
            C
            O
            N
            Cl

    Feature 5:
        Number of directly bonded neighboring atoms.

    Feature 6:
        Sum of bond orders around the atom.

    Feature 7:
        Approximate atomic mass divided by 40.

    The final node-feature matrix has shape:

        number_of_atoms x 7
    """

    number_of_nodes = len(
        atom_types
    )


    degree = np.zeros(
        number_of_nodes,
        dtype=np.float32
    )


    bond_order_sum = np.zeros(
        number_of_nodes,
        dtype=np.float32
    )


    for node_1, node_2, bond_order in bonds:

        degree[
            node_1
        ] += 1.0

        degree[
            node_2
        ] += 1.0


        bond_order_sum[
            node_1
        ] += bond_order

        bond_order_sum[
            node_2
        ] += bond_order


    feature_rows = []


    for node_index, atom_symbol in enumerate(
        atom_types
    ):

        one_hot_atom_type = [
            1.0 if atom_symbol == atom_type else 0.0
            for atom_type in ATOM_TYPES
        ]


        normalized_degree = (
            degree[node_index]
            / 4.0
        )


        normalized_bond_order = (
            bond_order_sum[node_index]
            / 4.0
        )


        normalized_mass = (
            ATOMIC_MASS[
                atom_symbol
            ]
            / 40.0
        )


        node_feature = (
            one_hot_atom_type
            + [
                normalized_degree,
                normalized_bond_order,
                normalized_mass,
            ]
        )


        feature_rows.append(
            node_feature
        )


    node_features = np.array(
        feature_rows,
        dtype=np.float32
    )


    return node_features


# ===========================================================================
# 10. GENERATE ONE MOLECULE GRAPH
# ===========================================================================

def generate_molecule(
    molecule_index
):
    """
    Generate one synthetic molecule and its graph representation.
    """

    family = random.choice(
        MOLECULE_FAMILIES
    )


    if family == "ketone":

        # A simple ketone requires at least three carbon atoms.

        number_of_carbons = random.randint(
            3,
            10
        )

    else:

        number_of_carbons = random.randint(
            2,
            10
        )


    # Branching is only allowed when enough carbons are present.

    if number_of_carbons >= 4:

        is_branched = (
            random.random()
            < 0.35
        )

    else:

        is_branched = False


    atom_types, bonds, positions = build_carbon_skeleton(
        number_of_carbons,
        is_branched
    )


    add_functional_group(
        atom_types,
        bonds,
        positions,
        family
    )


    node_features = create_node_features(
        atom_types,
        bonds
    )


    adjacency = create_adjacency_matrix(
        len(atom_types),
        bonds
    )


    boiling_temperature = synthetic_boiling_temperature(
        number_of_carbons,
        family,
        is_branched
    )


    formula = create_formula(
        number_of_carbons,
        family
    )


    molecule_id = (
        f"MOL_{molecule_index:04d}"
    )


    molecule = {
        "Molecule_ID":
            molecule_id,

        "Family":
            family,

        "Formula":
            formula,

        "Number_of_Carbons":
            number_of_carbons,

        "Is_Branched":
            is_branched,

        "Atom_Types":
            atom_types,

        "Bonds":
            bonds,

        "Positions":
            positions,

        "Node_Features":
            node_features,

        "Adjacency":
            adjacency,

        "Boiling_Temperature_K":
            boiling_temperature,
    }


    return molecule


# ===========================================================================
# 11. GENERATE THE FULL DATASET
# ===========================================================================

molecules = []


for molecule_index in range(
    NUMBER_OF_MOLECULES
):

    molecule = generate_molecule(
        molecule_index
    )

    molecules.append(
        molecule
    )


print(
    "Generated molecules:",
    len(molecules)
)
print()


# ===========================================================================
# 12. SAVE A READABLE MOLECULE TABLE
# ===========================================================================

dataset_records = []


for molecule in molecules:

    dataset_records.append(
        {
            "Molecule_ID":
                molecule[
                    "Molecule_ID"
                ],

            "Family":
                molecule[
                    "Family"
                ],

            "Formula":
                molecule[
                    "Formula"
                ],

            "Number_of_Carbons":
                molecule[
                    "Number_of_Carbons"
                ],

            "Is_Branched":
                molecule[
                    "Is_Branched"
                ],

            "Number_of_Graph_Nodes":
                len(
                    molecule[
                        "Atom_Types"
                    ]
                ),

            "Boiling_Temperature_K":
                molecule[
                    "Boiling_Temperature_K"
                ],
        }
    )


dataset_df = pd.DataFrame(
    dataset_records
)


dataset_df = dataset_df.round(
    3
)


dataset_file = (
    RESULTS_DIRECTORY
    / "synthetic_molecule_dataset.csv"
)


dataset_df.to_csv(
    dataset_file,
    index=False
)


print("Example dataset rows:")
print(dataset_df.head())
print()


# ===========================================================================
# 13. DRAW ONE EXAMPLE MOLECULAR GRAPH
# ===========================================================================

def draw_molecular_graph(
    molecule,
    output_file
):
    """
    Draw one molecular graph using only matplotlib.

    No graph library is required.
    """

    positions = molecule[
        "Positions"
    ]

    atom_types = molecule[
        "Atom_Types"
    ]

    bonds = molecule[
        "Bonds"
    ]


    plt.figure(
        figsize=(8, 4)
    )


    # -----------------------------------------------------------------------
    # Draw bonds first so that nodes appear on top.
    # -----------------------------------------------------------------------

    for node_1, node_2, bond_order in bonds:

        x1, y1 = positions[
            node_1
        ]

        x2, y2 = positions[
            node_2
        ]


        # Use a thicker line for higher bond order.

        line_width = (
            1.5
            + 0.8 * (
                bond_order - 1
            )
        )


        plt.plot(
            [x1, x2],
            [y1, y2],
            linewidth=line_width
        )


    # -----------------------------------------------------------------------
    # Draw atoms.
    # -----------------------------------------------------------------------

    for node_index, atom_symbol in enumerate(
        atom_types
    ):

        x, y = positions[
            node_index
        ]


        plt.scatter(
            x,
            y,
            s=650,
            edgecolors="black"
        )


        plt.text(
            x,
            y,
            atom_symbol,
            ha="center",
            va="center",
            fontsize=11
        )


    plt.title(
        molecule["Molecule_ID"]
        + " | "
        + molecule["Family"]
        + " | "
        + molecule["Formula"]
        + "\n"
        + f'Boiling T = {molecule["Boiling_Temperature_K"]:.1f} K'
    )


    plt.axis(
        "equal"
    )

    plt.axis(
        "off"
    )

    plt.tight_layout()


    plt.savefig(
        output_file,
        dpi=200
    )


    plt.show()


example_graph_file = (
    RESULTS_DIRECTORY
    / "01_example_molecular_graph.png"
)


draw_molecular_graph(
    molecules[0],
    example_graph_file
)


# ===========================================================================
# 14. DATASET OVERVIEW PLOT
# ===========================================================================

figure, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(12, 5)
)


# ---------------------------------------------------------------------------
# Left: carbon count versus boiling temperature
# ---------------------------------------------------------------------------

axes[0].scatter(
    dataset_df[
        "Number_of_Carbons"
    ],
    dataset_df[
        "Boiling_Temperature_K"
    ]
)


axes[0].set_xlabel(
    "Number of Carbon Atoms"
)

axes[0].set_ylabel(
    "Boiling Temperature (K)"
)

axes[0].set_title(
    "Molecule Size and Boiling Temperature"
)


# ---------------------------------------------------------------------------
# Right: boiling-temperature distribution by molecule family
# ---------------------------------------------------------------------------

family_data = []


for family in MOLECULE_FAMILIES:

    family_temperatures = dataset_df[
        dataset_df[
            "Family"
        ]
        == family
    ][
        "Boiling_Temperature_K"
    ].to_numpy()


    family_data.append(
        family_temperatures
    )


axes[1].boxplot(
    family_data,
    tick_labels=MOLECULE_FAMILIES
)


axes[1].set_ylabel(
    "Boiling Temperature (K)"
)

axes[1].set_title(
    "Boiling Temperature by Molecular Family"
)

axes[1].tick_params(
    axis="x",
    rotation=35
)


figure.suptitle(
    "Synthetic Molecular Dataset",
    fontsize=14
)


figure.tight_layout()


dataset_overview_file = (
    RESULTS_DIRECTORY
    / "02_dataset_overview.png"
)


figure.savefig(
    dataset_overview_file,
    dpi=200
)


plt.show()


# ===========================================================================
# 15. TRAIN / TEST SPLIT
# ===========================================================================

# We split molecule INDICES so that the graph dictionaries remain intact.

all_indices = np.arange(
    len(molecules)
)


all_families = dataset_df[
    "Family"
].to_numpy()


train_indices, test_indices = train_test_split(
    all_indices,
    test_size=TEST_FRACTION,
    random_state=RANDOM_SEED,
    stratify=all_families
)


training_molecules = [
    molecules[index]
    for index in train_indices
]


test_molecules = [
    molecules[index]
    for index in test_indices
]


print(
    "Training molecules:",
    len(training_molecules)
)

print(
    "Test molecules:",
    len(test_molecules)
)

print()


# ===========================================================================
# 16. CONVERT ONE MOLECULE TO PYTORCH TENSORS
# ===========================================================================

def molecule_to_tensors(
    molecule
):
    """
    Convert one molecular graph from NumPy arrays into PyTorch tensors.
    """

    node_features = torch.tensor(
        molecule[
            "Node_Features"
        ],
        dtype=torch.float32,
        device=DEVICE
    )


    adjacency = torch.tensor(
        molecule[
            "Adjacency"
        ],
        dtype=torch.float32,
        device=DEVICE
    )


    target = torch.tensor(
        [
            molecule[
                "Boiling_Temperature_K"
            ]
        ],
        dtype=torch.float32,
        device=DEVICE
    )


    return (
        node_features,
        adjacency,
        target
    )


# ===========================================================================
# 17. SIMPLE GRAPH-CONVOLUTION LAYER
# ===========================================================================

class SimpleGraphConvolution(
    nn.Module
):
    """
    One simple graph message-passing layer.

    For every atom:

        1. Look at the features of neighboring atoms.
        2. Average those neighboring features.
        3. Transform the atom's own features.
        4. Transform the neighboring features.
        5. Add the two pieces together.
        6. Apply a ReLU activation.

    This is a simplified example of message passing.
    """

    def __init__(
        self,
        input_size,
        output_size
    ):

        super().__init__()


        self.self_linear = nn.Linear(
            input_size,
            output_size
        )


        self.neighbor_linear = nn.Linear(
            input_size,
            output_size
        )


        self.activation = nn.ReLU()


    def forward(
        self,
        node_features,
        adjacency
    ):

        # -------------------------------------------------------------------
        # Count neighbors for every atom.
        # -------------------------------------------------------------------

        degree = adjacency.sum(
            dim=1,
            keepdim=True
        )


        # Prevent division by zero.

        degree = torch.clamp(
            degree,
            min=1.0
        )


        # -------------------------------------------------------------------
        # Gather messages from neighboring atoms.
        # -------------------------------------------------------------------

        neighbor_sum = (
            adjacency
            @ node_features
        )


        neighbor_mean = (
            neighbor_sum
            / degree
        )


        # -------------------------------------------------------------------
        # Transform self information and neighbor information.
        # -------------------------------------------------------------------

        self_information = self.self_linear(
            node_features
        )


        neighbor_information = self.neighbor_linear(
            neighbor_mean
        )


        updated_features = (
            self_information
            + neighbor_information
        )


        updated_features = self.activation(
            updated_features
        )


        return updated_features


# ===========================================================================
# 18. COMPLETE GRAPH NEURAL NETWORK
# ===========================================================================

class BoilingPointGNN(
    nn.Module
):
    """
    Small graph neural network for molecular boiling-temperature regression.

    Architecture
    ------------

    Atomic node features
            |
            v
    Graph convolution 1
            |
            v
    Graph convolution 2
            |
            v
    Mean graph pooling
            |
            v
    Fully connected layer
            |
            v
    One boiling-temperature prediction
    """

    def __init__(
        self
    ):

        super().__init__()


        # Seven features are created for every atom.

        self.graph_layer_1 = SimpleGraphConvolution(
            input_size=7,
            output_size=32
        )


        self.graph_layer_2 = SimpleGraphConvolution(
            input_size=32,
            output_size=32
        )


        self.regression_head = nn.Sequential(

            nn.Linear(
                32,
                16
            ),

            nn.ReLU(),

            nn.Linear(
                16,
                1
            ),
        )


    def forward(
        self,
        node_features,
        adjacency
    ):

        # -------------------------------------------------------------------
        # Message passing
        # -------------------------------------------------------------------

        node_features = self.graph_layer_1(
            node_features,
            adjacency
        )


        node_features = self.graph_layer_2(
            node_features,
            adjacency
        )


        # -------------------------------------------------------------------
        # Graph pooling
        # -------------------------------------------------------------------

        # Every molecule has a different number of atoms.
        #
        # Sum pooling converts the variable number of atomic feature vectors
        # into ONE fixed-length molecular feature vector.
        #
        # Sum pooling is useful here because it preserves information about
        # how many atoms are present in the molecule.

        graph_feature_vector = node_features.sum(
            dim=0
        )


        # -------------------------------------------------------------------
        # Regression
        # -------------------------------------------------------------------

        predicted_boiling_temperature = self.regression_head(
            graph_feature_vector
        )


        return predicted_boiling_temperature


model = BoilingPointGNN().to(
    DEVICE
)


print("=" * 75)
print("GNN architecture")
print("=" * 75)

print(
    model
)

print()


number_of_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)


print(
    "Trainable parameters:",
    number_of_parameters
)

print()


# ===========================================================================
# 19. LOSS FUNCTION AND OPTIMIZER
# ===========================================================================

loss_function = nn.MSELoss()


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ===========================================================================
# 20. TRAIN THE GNN
# ===========================================================================

training_history = []


for epoch in range(
    1,
    NUMBER_OF_EPOCHS + 1
):

    model.train()


    # Shuffle molecule order every epoch.

    random.shuffle(
        training_molecules
    )


    epoch_losses = []


    for molecule in training_molecules:

        node_features, adjacency, target = molecule_to_tensors(
            molecule
        )


        # ---------------------------------------------------------------
        # Clear gradients from the previous molecule.
        # ---------------------------------------------------------------

        optimizer.zero_grad()


        # ---------------------------------------------------------------
        # Forward propagation through the graph neural network.
        # ---------------------------------------------------------------

        prediction = model(
            node_features,
            adjacency
        )


        # ---------------------------------------------------------------
        # Calculate regression loss.
        # ---------------------------------------------------------------

        loss = loss_function(
            prediction,
            target
        )


        # ---------------------------------------------------------------
        # Backpropagation.
        # ---------------------------------------------------------------

        loss.backward()


        # ---------------------------------------------------------------
        # Update the model parameters.
        # ---------------------------------------------------------------

        optimizer.step()


        epoch_losses.append(
            loss.item()
        )


    # -----------------------------------------------------------------------
    # Calculate training-set predictions once per epoch.
    # -----------------------------------------------------------------------

    model.eval()


    epoch_true_temperatures = []

    epoch_predicted_temperatures = []


    with torch.no_grad():

        for molecule in training_molecules:

            node_features, adjacency, target = molecule_to_tensors(
                molecule
            )


            prediction = model(
                node_features,
                adjacency
            )


            epoch_true_temperatures.append(
                target.item()
            )


            epoch_predicted_temperatures.append(
                prediction.item()
            )


    epoch_mae = mean_absolute_error(
        epoch_true_temperatures,
        epoch_predicted_temperatures
    )


    epoch_r2 = r2_score(
        epoch_true_temperatures,
        epoch_predicted_temperatures
    )


    average_loss = np.mean(
        epoch_losses
    )


    training_history.append(
        {
            "Epoch":
                epoch,

            "Average_Training_MSE":
                average_loss,

            "Training_MAE_K":
                epoch_mae,

            "Training_R2":
                epoch_r2,
        }
    )


    if (
        epoch == 1
        or epoch % PRINT_EVERY == 0
        or epoch == NUMBER_OF_EPOCHS
    ):

        print(
            f"Epoch {epoch:3d} | "
            f"MSE = {average_loss:9.2f} | "
            f"MAE = {epoch_mae:7.2f} K | "
            f"R2 = {epoch_r2:6.3f}"
        )


print()


# ===========================================================================
# 21. SAVE AND PLOT TRAINING HISTORY
# ===========================================================================

training_history_df = pd.DataFrame(
    training_history
)


training_history_file = (
    RESULTS_DIRECTORY
    / "gnn_training_history.csv"
)


training_history_df.to_csv(
    training_history_file,
    index=False
)


figure, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(12, 5)
)


axes[0].plot(
    training_history_df[
        "Epoch"
    ],
    training_history_df[
        "Average_Training_MSE"
    ]
)


axes[0].set_xlabel(
    "Epoch"
)

axes[0].set_ylabel(
    "Average MSE Loss"
)

axes[0].set_title(
    "GNN Training Loss"
)


axes[1].plot(
    training_history_df[
        "Epoch"
    ],
    training_history_df[
        "Training_MAE_K"
    ]
)


axes[1].set_xlabel(
    "Epoch"
)

axes[1].set_ylabel(
    "Training MAE (K)"
)

axes[1].set_title(
    "GNN Training Error"
)


figure.suptitle(
    "GNN Training Diagnostics",
    fontsize=14
)


figure.tight_layout()


training_plot_file = (
    RESULTS_DIRECTORY
    / "03_gnn_training_diagnostics.png"
)


figure.savefig(
    training_plot_file,
    dpi=200
)


plt.show()


# ===========================================================================
# 22. PREDICTION HELPER FUNCTION
# ===========================================================================

def predict_molecule_list(
    model,
    molecule_list
):
    """
    Predict boiling temperatures for a list of molecular graphs.
    """

    model.eval()


    true_temperatures = []

    predicted_temperatures = []

    molecule_ids = []


    with torch.no_grad():

        for molecule in molecule_list:

            node_features, adjacency, target = molecule_to_tensors(
                molecule
            )


            prediction = model(
                node_features,
                adjacency
            )


            molecule_ids.append(
                molecule[
                    "Molecule_ID"
                ]
            )


            true_temperatures.append(
                target.item()
            )


            predicted_temperatures.append(
                prediction.item()
            )


    return (
        molecule_ids,
        np.array(
            true_temperatures
        ),
        np.array(
            predicted_temperatures
        )
    )


# ===========================================================================
# 23. FINAL TRAINING AND TEST PREDICTIONS
# ===========================================================================

(
    training_ids,
    y_train,
    train_predictions
) = predict_molecule_list(
    model,
    training_molecules
)


(
    test_ids,
    y_test,
    test_predictions
) = predict_molecule_list(
    model,
    test_molecules
)


# ===========================================================================
# 24. FINAL PERFORMANCE METRICS
# ===========================================================================

train_mae = mean_absolute_error(
    y_train,
    train_predictions
)


train_rmse = np.sqrt(
    mean_squared_error(
        y_train,
        train_predictions
    )
)


train_r2 = r2_score(
    y_train,
    train_predictions
)


test_mae = mean_absolute_error(
    y_test,
    test_predictions
)


test_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        test_predictions
    )
)


test_r2 = r2_score(
    y_test,
    test_predictions
)


print("=" * 75)
print("Final GNN performance")
print("=" * 75)


print(
    f"Training: "
    f"MAE = {train_mae:.1f} K, "
    f"RMSE = {train_rmse:.1f} K, "
    f"R2 = {train_r2:.3f}"
)


print(
    f"Test:     "
    f"MAE = {test_mae:.1f} K, "
    f"RMSE = {test_rmse:.1f} K, "
    f"R2 = {test_r2:.3f}"
)


print()


# ===========================================================================
# 25. SAVE PREDICTION TABLES
# ===========================================================================

training_predictions_df = pd.DataFrame(
    {
        "Molecule_ID":
            training_ids,

        "True_Boiling_Temperature_K":
            y_train,

        "GNN_Predicted_Boiling_Temperature_K":
            train_predictions,

        "Absolute_Error_K":
            np.abs(
                train_predictions
                - y_train
            ),
    }
)


test_predictions_df = pd.DataFrame(
    {
        "Molecule_ID":
            test_ids,

        "True_Boiling_Temperature_K":
            y_test,

        "GNN_Predicted_Boiling_Temperature_K":
            test_predictions,

        "Absolute_Error_K":
            np.abs(
                test_predictions
                - y_test
            ),
    }
)


training_predictions_df.round(
    3
).to_csv(
    RESULTS_DIRECTORY
    / "gnn_training_predictions.csv",
    index=False
)


test_predictions_df.round(
    3
).to_csv(
    RESULTS_DIRECTORY
    / "gnn_test_predictions.csv",
    index=False
)


# ===========================================================================
# 26. TRAINING AND TEST PARITY PLOTS
# ===========================================================================

all_temperature_values = np.concatenate(
    [
        y_train,
        train_predictions,
        y_test,
        test_predictions,
    ]
)


plot_minimum = (
    np.min(
        all_temperature_values
    )
    - 20.0
)


plot_maximum = (
    np.max(
        all_temperature_values
    )
    + 20.0
)


figure, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(12, 5)
)


# ---------------------------------------------------------------------------
# Training parity
# ---------------------------------------------------------------------------

axes[0].scatter(
    y_train,
    train_predictions
)


axes[0].plot(
    [
        plot_minimum,
        plot_maximum
    ],
    [
        plot_minimum,
        plot_maximum
    ],
    linestyle="--"
)


axes[0].set_xlim(
    plot_minimum,
    plot_maximum
)

axes[0].set_ylim(
    plot_minimum,
    plot_maximum
)


axes[0].set_xlabel(
    "True Boiling Temperature (K)"
)

axes[0].set_ylabel(
    "Predicted Boiling Temperature (K)"
)


axes[0].set_title(
    "Training Set"
    + f"\nR2 = {train_r2:.3f}, "
    + f"MAE = {train_mae:.1f} K"
)


# ---------------------------------------------------------------------------
# Test parity
# ---------------------------------------------------------------------------

axes[1].scatter(
    y_test,
    test_predictions
)


axes[1].plot(
    [
        plot_minimum,
        plot_maximum
    ],
    [
        plot_minimum,
        plot_maximum
    ],
    linestyle="--"
)


axes[1].set_xlim(
    plot_minimum,
    plot_maximum
)

axes[1].set_ylim(
    plot_minimum,
    plot_maximum
)


axes[1].set_xlabel(
    "True Boiling Temperature (K)"
)

axes[1].set_ylabel(
    "Predicted Boiling Temperature (K)"
)


axes[1].set_title(
    "Test Set"
    + f"\nR2 = {test_r2:.3f}, "
    + f"MAE = {test_mae:.1f} K"
)


figure.suptitle(
    "GNN Boiling-Temperature Regression",
    fontsize=14
)


figure.tight_layout()


parity_plot_file = (
    RESULTS_DIRECTORY
    / "04_gnn_boiling_temperature_parity.png"
)


figure.savefig(
    parity_plot_file,
    dpi=200
)


plt.show()


# ===========================================================================
# 27. FINAL OUTPUT SUMMARY
# ===========================================================================

print("Analysis complete.")
print()

print("Main output files:")
print()

print(
    "Synthetic molecule table:"
)
print(
    dataset_file
)
print()

print(
    "Example molecular graph:"
)
print(
    example_graph_file
)
print()

print(
    "Training diagnostics:"
)
print(
    training_plot_file
)
print()

print(
    "Training/test parity plots:"
)
print(
    parity_plot_file
)
print()


# ===========================================================================
# PRACTICE EXERCISES
# ===========================================================================

# 1. Change NUMBER_OF_MOLECULES from 300 to 100.
#    How does the test accuracy change?
#
# 2. Change the first graph layer from 32 hidden features to 16.
#    Does the smaller GNN perform differently?
#
# 3. Add a third SimpleGraphConvolution layer.
#    Does a deeper GNN improve the result?
#
# 4. Change the graph pooling operation from:
#
#        node_features.mean(dim=0)
#
#    to:
#
#        node_features.sum(dim=0)
#
#    What changes?
#
# 5. Remove atomic mass from create_node_features().
#    Does the model still distinguish chlorine-containing molecules well?
#
# 6. Add another molecule family to the synthetic generator.
#
# 7. Increase the random noise inside synthetic_boiling_temperature().
#    How does this affect the maximum prediction accuracy?
#
# 8. Compare linear and branched molecules.
#    Can the GNN learn that branching changes boiling temperature?
#
# 9. Examine one adjacency matrix by printing:
#
#        molecules[0]["Adjacency"]
#
#    How does this matrix represent the bonds in the molecule?
#
# 10. Examine the node-feature matrix:
#
#        molecules[0]["Node_Features"]
#
#     How is this different from the single material feature vectors used in
#     the earlier regression and neural-network examples?
