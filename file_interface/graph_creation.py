import os
import random
import networkx as nx
import osmnx as ox


def load_graph():
    # Bounding box for Benguet, La Union, Pangasinan
    north = 16.762507
    south = 15.718278
    east  = 120.872237
    west  = 119.665801
    bbox = (west, south, east, north)

    random_seed = 42
    output_dir = "output"
    graph_file = os.path.join(output_dir, "road_network.graphml")

    os.makedirs(output_dir, exist_ok=True)
    random.seed(random_seed)

    ox.settings.use_cache = True
    ox.settings.log_console = True

    if os.path.exists(graph_file):
        print(f"Loading graph from {graph_file} ...")
        G = ox.load_graphml(graph_file)
        print("Graph loaded.")
    else:
        # Download the graph using bbox
        print("Downloading graph from OpenStreetMap ...")
        G = ox.graph_from_bbox(
            bbox,
            network_type="drive",
            simplify=True
        )
        # Convert to undirected
        G = G.to_undirected()
        ox.save_graphml(G, graph_file)
        print(f"Graph saved to {graph_file}")

    print(f"Nodes = {len(G.nodes())}")
    print(f"Edges = {len(G.edges())}")
    print(nx.is_connected(G))
    return G