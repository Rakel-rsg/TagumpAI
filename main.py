import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import networkx as nx
from file_interface import *
from solvers import *


def plot_routes(routes, origin_of, farmers, office_node, positions):
    """Plot the node graph and technician routes using NetworkX.

    Parameters
    ----------
    routes      : dict  technician_id -> list of node IDs (ordered path)
    origin_of   : dict  technician_id -> home node ID
    farmers     : list  of farmer node IDs
    office_node : node ID of the office
    positions   : dict  node_id -> (x, y)  — 2-D coordinates for drawing
    """
    G = nx.DiGraph()
    G.add_nodes_from(positions)

    tech_origins = set(origin_of.values())
    farmer_set   = set(farmers)

    node_colors = []
    for node in G.nodes():
        if node == office_node:
            node_colors.append("gold")
        elif node in tech_origins:
            node_colors.append("cornflowerblue")
        elif node in farmer_set:
            node_colors.append("lightgreen")
        else:
            node_colors.append("lightgray")

    palette = ["tab:red", "tab:blue", "tab:purple", "tab:orange", "tab:cyan"]
    route_colors = {t: palette[i % len(palette)] for i, t in enumerate(routes)}

    for t, path in routes.items():
        color = route_colors[t]
        for a, b in zip(path, path[1:]):
            G.add_edge(a, b, color=color)

    edge_colors = [G[u][v]["color"] for u, v in G.edges()]

    node_labels = {}
    for node in G.nodes():
        if node == office_node:
            node_labels[node] = f"{node}\n(O)"
        elif node in tech_origins:
            tech_id = next(t for t, n in origin_of.items() if n == node)
            node_labels[node] = f"{node}\n({tech_id})"
        else:
            node_labels[node] = str(node)

    _, ax = plt.subplots(figsize=(9, 7))

    nx.draw_networkx_nodes(G, pos=positions, node_color=node_colors,
                           node_size=900, ax=ax)
    nx.draw_networkx_labels(G, pos=positions, labels=node_labels,
                            font_size=8, ax=ax)
    nx.draw_networkx_edges(G, pos=positions, edge_color=edge_colors,
                           arrows=True, arrowsize=20,
                           connectionstyle="arc3,rad=0.1", ax=ax)

    legend_handles = [
        mpatches.Patch(color="gold",           label="Office"),
        mpatches.Patch(color="cornflowerblue", label="Technician home"),
        mpatches.Patch(color="lightgreen",     label="Farmer"),
    ] + [
        Line2D([0], [0], color=c, linewidth=2, label=t)
        for t, c in route_colors.items()
    ]
    ax.legend(handles=legend_handles, loc="best")
    ax.set_title("Technician Routes")
    ax.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # --- Simple hand-checkable instance ---
    # (all_nodes, farmer_nodes, technicians, office_node,
    #  initial_straws, distance, origin_of,
    #  positions, total_straws_available) = instance_creation_simple()

    # routes = solve_technician_routing(
    #     all_nodes=all_nodes,
    #     farmers=farmer_nodes,
    #     technicians=technicians,
    #     office_node=office_node,
    #     initial_straws=initial_straws,
    #     total_straws_available=total_straws_available,
    #     full_distance=distance,
    #     origin_of=origin_of,
    #     verbose=True
    # )

    # print("\nRoutes:")
    # for t, path in routes.items():
    #     print(f"  {t}: {' -> '.join(str(n) for n in path)}")

    # plot_routes(routes, origin_of, farmer_nodes, office_node, positions)

    # --- Full real-world instance (uncomment to run) ---
    all_nodes_ordered, farmer_nodes, technicians, office_node, initial_straws_new, distance_matrix, origin_of = instance_creation()
    total_straws_available = 200
    routes = solve_technician_routing(
        all_nodes=all_nodes_ordered,
        farmers=farmer_nodes,
        technicians=technicians,
        office_node=office_node,
        initial_straws=initial_straws_new,
        total_straws_available=total_straws_available,
        full_distance=distance_matrix,
        origin_of=origin_of,
    )
