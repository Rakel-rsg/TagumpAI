import gurobipy as gp
from gurobipy import GRB
import os


def solve_technician_routing(
    all_nodes,
    farmers,
    technicians,
    office_node,
    initial_straws,
    full_distance,
    origin_of,
    hexaly_x=None,     
    hexaly_r=None,     
    time_limit=None,
    verbose=False
):

    all_nodes = sorted(set(all_nodes))
    farmers = sorted(set(farmers))

    index_map = {node: idx for idx, node in enumerate(all_nodes)}

    m = gp.Model("TechnicianRouting")

    if time_limit is not None:
        m.Params.TimeLimit = time_limit
    if verbose:
        m.Params.OutputFlag = 1

    # -------------------------
    # VARIABLES
    # -------------------------
    x = m.addVars(all_nodes, all_nodes, technicians,
                  vtype=GRB.BINARY, name="x")

    r = m.addVars(all_nodes, technicians,
                  lb=0, ub=10,
                  vtype=GRB.INTEGER, name="r")

    # -------------------------
    # WARM START FROM HEXALY
    # -------------------------
    if hexaly_x is not None:
        print("Applying Hexaly warm start...")

        for t in technicians:
            for i in all_nodes:
                for j in all_nodes:
                    if i == j:
                        continue
                    if hexaly_x[t][i][j] > 0.5:
                        x[i, j, t].Start = 1.0
                    else:
                        x[i, j, t].Start = 0.0

    if hexaly_r is not None:
        for t in technicians:
            for i in all_nodes:
                if i in hexaly_r[t]:
                    r[i, t].Start = hexaly_r[t][i]

    # -------------------------
    # DISTANCE
    # -------------------------
    def dist(i, j):
        if isinstance(full_distance, dict):
            return full_distance[i][j]
        return full_distance[index_map[i]][index_map[j]]

    # -------------------------
    # OBJECTIVE
    # -------------------------
    m.setObjective(
        gp.quicksum(
            dist(i, j) * x[i, j, t]
            for t in technicians
            for i in all_nodes
            for j in all_nodes
            if i != j
        ),
        GRB.MINIMIZE
    )

    # -------------------------
    # ROUTING CONSTRAINTS
    # -------------------------
    for t in technicians:
        origin = origin_of[t]

        m.addConstr(
            gp.quicksum(x[origin, j, t] for j in all_nodes if j != origin) == 1
        )

        m.addConstr(
            gp.quicksum(x[i, origin, t] for i in all_nodes if i != origin) == 1
        )

        for j in all_nodes:
            m.addConstr(
                gp.quicksum(x[i, j, t] for i in all_nodes if i != j) ==
                gp.quicksum(x[j, k, t] for k in all_nodes if k != j)
            )

    # each farmer visited once
    for f in farmers:
        m.addConstr(
            gp.quicksum(x[i, f, t]
                        for t in technicians
                        for i in all_nodes if i != f) == 1
        )


    for t in technicians:
        origin = origin_of[t]

        m.addConstr(r[origin, t] == initial_straws[t])

        for i in all_nodes:
            for j in all_nodes:

                if i == j:
                    continue

                if j in farmers:
                    m.addConstr(
                        r[j, t] >= r[i, t] - 1 - 1e6 * (1 - x[i, j, t])
                    )
                    m.addConstr(
                        r[j, t] <= r[i, t] - 1 + 1e6 * (1 - x[i, j, t])
                    )

                if j == office_node:
                    m.addConstr(
                        r[j, t] >= r[i, t] + 1e6 * (x[i, j, t]) - 1e6
                    )
                    m.addConstr(
                        r[j, t] <= r[i, t] + 1e6 * x[i, j, t]
                    )

    # -------------------------
    # SUBTOUR ELIMINATION (lazy)
    # -------------------------
    m.Params.LazyConstraints = 1

    def subtour_callback(model, where):
        if where != GRB.Callback.MIPSOL:
            return

        x_sol = model.cbGetSolution(x)

        for t in technicians:
            succ = {
                i: j
                for i in all_nodes
                for j in all_nodes
                if i != j and x_sol[i, j, t] > 0.5
            }

            visited = set()
            for start in all_nodes:
                if start in visited:
                    continue

                cycle = []
                cur = start

                while cur not in visited and cur in succ:
                    visited.add(cur)
                    cycle.append(cur)
                    cur = succ[cur]

                if len(cycle) > 1 and office_node not in cycle:
                    model.cbLazy(
                        gp.quicksum(
                            x[i, j, t]
                            for i in cycle
                            for j in cycle
                            if i != j
                        ) <= len(cycle) - 1
                    )

    # -------------------------
    # OPTIMIZE
    # -------------------------
    m.optimize(subtour_callback)

    m.write(os.path.join("logs", "model.lp"))

    # -------------------------
    # EXTRACT
    # -------------------------
    routes = {}
    arcs = {t: [] for t in technicians}

    if m.SolCount > 0:
        x_vals = m.getAttr("X", x)

        for (i, j, t), val in x_vals.items():
            if val > 0.5:
                arcs[t].append((i, j))

        for t in technicians:
            origin = origin_of[t]

            succ = {
                i: j
                for (i, j, tt), val in x_vals.items()
                if tt == t and val > 0.5
            }

            path = [origin]
            cur = origin
            seen = {origin}

            while cur in succ:
                nxt = succ[cur]
                if nxt in seen:
                    break
                path.append(nxt)
                seen.add(nxt)
                cur = nxt

            routes[t] = path

    return routes, arcs