import os
import math
import pandas as pd
import numpy as np
import networkx as nx
import osmnx as ox
from .graph_creation import load_graph


def instance_creation():
    G = load_graph()
    office_lat, office_lon = 16.611823, 120.310104
    print(f"Setting office at PCC-UPLB coordinates: ({office_lat}, {office_lon})")
    # Read farmer coordinates from Excel
    farmer_df = pd.read_excel(os.path.join("data", "farmers.xlsx"))
    techs_df = pd.read_excel(os.path.join("data", "technicians.xlsx"))

    # Build the GeoDataFrame and BallTree once by passing all query points
    # in a single vectorised call instead of one call per row.
    all_lons = [office_lon] + list(farmer_df['Longitude']) + list(techs_df['Longitude'])
    all_lats = [office_lat] + list(farmer_df['Latitude'])  + list(techs_df['Latitude'])
    all_nearest = ox.distance.nearest_nodes(G, X=all_lons, Y=all_lats)

    n_farmers = len(farmer_df)
    office_node       = all_nearest[0]
    farmer_nodes_raw  = list(all_nearest[1 : 1 + n_farmers])
    technician_nodes  = list(all_nearest[1 + n_farmers :])
    print("Nearest node to PCC-UPLB:", office_node)

    # Remove duplicates
    technician_nodes = list(set(technician_nodes))

    # Remove farmers that are on the same node as any technician
    farmer_nodes = [n for n in farmer_nodes_raw if n not in technician_nodes]

    # Remove duplicates in remaining farmers
    farmer_nodes = list(set(farmer_nodes))

    print(f"Selected {len(farmer_nodes)} farmers, {len(technician_nodes)} technicians.")

    all_nodes_ordered = technician_nodes + farmer_nodes + [office_node]
    index_by_node = {node: i for i, node in enumerate(all_nodes_ordered)}
    n_nodes = len(all_nodes_ordered)

    matrix_file = os.path.join('output', "full_distance_matrix.csv")

    distance_matrix = None
    if os.path.exists(matrix_file):
        print(f"Loading distance matrix from {matrix_file} ...")
        df_cached = pd.read_csv(matrix_file, index_col=0)
        df_cached.index   = df_cached.index.astype(int)
        df_cached.columns = df_cached.columns.astype(int)
        if all(n in df_cached.index for n in all_nodes_ordered):
            distance_matrix = df_cached.loc[all_nodes_ordered, all_nodes_ordered].values
            print("Distance matrix loaded.")
        else:
            print("Cached matrix does not cover current nodes — recomputing ...")

    if distance_matrix is None:
        print("Computing full all-pairs shortest path distance matrix ...")
        distance_matrix = np.zeros((n_nodes, n_nodes), dtype=float)
        for i, src_node in enumerate(all_nodes_ordered):
            lengths = nx.single_source_dijkstra_path_length(G, src_node, weight="length")
            for j, tgt in enumerate(all_nodes_ordered):
                distance_matrix[i, j] = lengths.get(tgt, float("inf"))
        distance_matrix[np.isinf(distance_matrix)] = 1e6
        df_full = pd.DataFrame(distance_matrix, index=all_nodes_ordered, columns=all_nodes_ordered)
        df_full.to_csv(matrix_file)
        print(f"Distance matrix saved to {matrix_file}")

    # Extract distance matrix for optimization:
    n_t = len(technician_nodes)
    n_f = len(farmer_nodes)
    idx_tech_start = 0
    idx_far_start = n_t
    idx_office = n_nodes - 1

    distance_ft = distance_matrix[idx_far_start:idx_far_start + n_f, idx_tech_start:idx_tech_start + n_t]

    # office to tech, office to farmer
    office_to_tech = distance_matrix[idx_office, idx_tech_start:idx_tech_start + n_t]
    office_to_far = distance_matrix[idx_office, idx_far_start:idx_far_start + n_f]
    tech_to_office = distance_matrix[idx_tech_start:idx_tech_start + n_t, idx_office]
    far_to_office = distance_matrix[idx_far_start:idx_far_start + n_f, idx_office]

    # Checking
    print(f"Sample distance: ")
    print(f"   tech0 -> farm0 = {distance_ft[5,5]:.2f} m")
    print(f"   Office -> tech0 = {office_to_tech[5]:.2f} m")
    print(f"   Office -> farm0 = {far_to_office[0]:.2f} m")

    tech_limits = {t: 40 for t in technician_nodes}
    carrying_straws = {t: 5 for t in technician_nodes}
    total_straws_available = 200

    farmer_preferences = {}

    tech_idx_to_node = {j: technician_nodes[j] for j in range(n_t)}
    far_idx_to_node = {i: farmer_nodes[i] for i in range(n_f)}
    tech_node_to_idx = {v: k for k, v in tech_idx_to_node.items()}
    far_node_to_idx = {v: k for k, v in far_idx_to_node.items()}
    tech_limits = {t: 40 for t in technician_nodes}

    # Create technician IDs
    technicians = [f"T{i}" for i in range(len(technician_nodes))]

    # Map technician ID -> graph node
    origin_of = {
        tech: node
        for tech, node in zip(technicians, technician_nodes)
    }

    # Ensure all_nodes contains unique graph nodes
    all_nodes_ordered = sorted(
        set(farmer_nodes)
        .union(technician_nodes)
        .union([office_node])
    )

    # Initial straws per technician (from assignments)
    initial_straws = {t: 3 for t in technician_nodes}

    initial_straws_new = {
        tech: initial_straws[node]
        for tech, node in origin_of.items()
    }
    return all_nodes_ordered, farmer_nodes, technicians, office_node, initial_straws_new, distance_matrix, origin_of


def instance_creation_simple():
    """
    Tiny hand-checkable instance.

    Layout (x, y):
        F0(0,3)  F1(2,3)  F2(4,3)
        T0(0,1)  O(2,0)   T1(4,1)

    Node IDs:  0=office, 1=T0 home, 2=T1 home, 3=F0, 4=F1, 5=F2

    Key distances (Euclidean, rounded for reference):
        T0->F0: 2.00   T0->F1: 2.83   T0->F2: 4.47
        T1->F0: 4.47   T1->F1: 2.83   T1->F2: 2.00
        F0->F1: 2.00   F1->F2: 2.00

    Each technician starts with 2 straws (enough to serve 2 farmers each).
    Optimal: T0 serves {F0, F1} or {F0}, T1 serves {F2} or {F1, F2} — total ~10.83.
    """
    positions = {
        0: (2, 0),  # office
        1: (0, 1),  # T0 home
        2: (4, 1),  # T1 home
        3: (0, 3),  # farmer F0
        4: (2, 3),  # farmer F1
        5: (4, 3),  # farmer F2
    }

    nodes = list(positions)
    distance = {
        i: {j: math.sqrt((positions[i][0] - positions[j][0]) ** 2 +
                          (positions[i][1] - positions[j][1]) ** 2)
            for j in nodes}
        for i in nodes
    }

    technicians = ["T0", "T1"]
    origin_of   = {"T0": 1, "T1": 2}
    farmer_nodes = [3, 4, 5]
    office_node  = 0
    initial_straws = {"T0": 1, "T1": 1}
    all_nodes = sorted(set(farmer_nodes) | set(origin_of.values()) | {office_node})
    total_straws_available = 10

    return all_nodes, farmer_nodes, technicians, office_node, initial_straws, distance, origin_of, positions, total_straws_available