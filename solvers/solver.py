import gurobipy as gp
from gurobipy import GRB
import os


def _find_subtours(succ, all_nodes, origin):
    """
    Detect directed cycles in `succ` (node -> successor) that do NOT contain
    `origin`.  Returns a list of sets, each being the nodes of one subtour.

    Because flow conservation forces in-degree == out-degree at every node,
    the active arcs form disjoint simple paths and simple cycles.  We exploit
    this structure: follow each unvisited node's successor chain and check
    whether we close a loop before reaching a node we already processed.
    """
    subtours = []
    visited  = set()

    for start in all_nodes:
        if start in visited:
            continue

        path       = []
        path_index = {}   # node -> position in `path`
        current    = start

        while current not in visited:
            if current in path_index:
                # closed a cycle
                cycle = path[path_index[current]:]
                if origin not in cycle:
                    subtours.append(set(cycle))
                break
            path_index[current] = len(path)
            path.append(current)
            current = succ.get(current)   # None if no outgoing arc
            if current is None:
                break

        visited.update(path)

    return subtours


def solve_technician_routing(
    all_nodes,
    farmers,
    technicians,
    office_node,
    initial_straws,
    total_straws_available,
    full_distance,
    origin_of,
    time_limit=None,
    verbose=False
):

    # Ensure uniqueness
    all_nodes = sorted(set(all_nodes))
    farmers = sorted(set(farmers))

    assert len(all_nodes) == len(set(all_nodes))
    assert len(farmers) == len(set(farmers))

    index_map = {node: idx for idx, node in enumerate(all_nodes)}
    N = len(all_nodes)
    big_M = 10**6

    m = gp.Model("TechnicianRouting")

    if time_limit is not None:
        m.Params.TimeLimit = time_limit
    if verbose:
        m.Params.OutputFlag = 1

    # x = 1 if technician travel to one node to another
    x = m.addVars(
        all_nodes,
        all_nodes,
        technicians,
        vtype=GRB.BINARY,
        name="x"
    )
    # number of straws that each technician has in all the nodes
    r = m.addVars(
        all_nodes,
        technicians,
        lb=0,
        ub=10, # TODO:update this upper bound on the number of straws
        vtype=GRB.INTEGER,
        name="r"
    )
    # center_visit = 1 if the technician visit the center.
    # visits = m.addVars(
    #     technicians,
    #     lb=0,
    #     ub=N,
    #     vtype=GRB.INTEGER,
    #     name="visits"
    # )

    # u = m.addVars(
    #     all_nodes,
    #     technicians,
    #     lb=0,
    #     ub=N - 1,
    #     vtype=GRB.INTEGER,
    #     name="u"
    # )

    # Refill amount per office visit — decision variable
    RA = m.addVars(
        technicians,
        lb=0,
        ub=total_straws_available,
        vtype=GRB.INTEGER,
        name="refill_amount"
    )

    # Auxiliary variable: q[t] = RA[t] * visits[t]  (linearised product)
    # q = m.addVars(
    #     technicians,
    #     lb=0,
    #     ub=total_straws_available * N,
    #     vtype=GRB.INTEGER,
    #     name="q"
    # )

    # Distance function
    def dist(i, j):
        if isinstance(full_distance, dict):
            return full_distance[i][j]
        return full_distance[index_map[i]][index_map[j]]

    # Objective
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


    # 1. START / END AT TECHNICIAN ORIGIN
    for t in technicians:
        origin = origin_of[t]
        m.addConstr(
            gp.quicksum(
                x[origin, j, t]
                for j in all_nodes
                if j != origin
            ) == 1,
            name=f"Start_{t}"
        )
        m.addConstr(
            gp.quicksum(
                x[i, origin, t]
                for i in all_nodes
                if i != origin
            ) == 1,
            name=f"Return_{t}"
        )

    # 2. FLOW CONSERVATION
    for t in technicians:
        for j in all_nodes:
            m.addConstr(
                gp.quicksum(
                    x[i, j, t]
                    for i in all_nodes
                    if i != j
                )
                ==
                gp.quicksum(
                    x[j, k, t]
                    for k in all_nodes
                    if k != j
                ),
                name=f"Flow_{j}_{t}"
            )


    # 3. EACH FARMER VISITED ONCE
    for f in farmers:
        m.addConstr(
            gp.quicksum(
                x[i, f, t]
                for t in technicians
                for i in all_nodes
                if i != f
            ) == 1,
            name=f"VisitFarmer_{f}"
        )

    # 4. STRAW INVENTORY
    for t in technicians:
        origin = origin_of[t]
        m.addConstr(
            r[origin, t] == initial_straws[t],
            name=f"InitialStraws_{t}"
        )

        for i in all_nodes:
            for j in all_nodes:
                
                if i == j:
                    continue

                if j in farmers:

                    m.addConstr(
                        r[j, t]
                        >=
                        r[i, t] - 1
                        - big_M * (1 - x[i, j, t])
                    )

                    m.addConstr(
                        r[j, t]
                        <=
                        r[i, t] - 1
                        + big_M * (1 - x[i, j, t])
                    )

                if j == office_node:

                    m.addConstr(
                        r[j, t]
                        >=
                        r[i, t]
                        + RA[t]
                        - big_M * (1 - x[i, j, t])
                    )

                    m.addConstr(
                        r[j, t] <= r[i, t] + RA[t] + big_M * (1 - x[i, j, t])
                    )

    # Global straw budget: total straws distributed cannot exceed supply
    m.addConstr(
        gp.quicksum(RA[t] for t in technicians) <= total_straws_available,
        name="GlobalStrawBudget"
    )

    # 4B. MUST HAVE STRAWS TO SERVE
    for t in technicians:
        for i in all_nodes:

            m.addConstr(
                gp.quicksum(
                    x[i, j, t]
                    for j in farmers
                    if j != i
                )
                <= r[i, t]
            )

    # # 5. OFFICE VISITS
    # for t in technicians:
    #     m.addConstr(
    #         visits[t]
    #         ==
    #         gp.quicksum(
    #             x[i, office_node, t]
    #             for i in all_nodes
    #             if i != office_node
    #         )
    #     )
    # # 6. McCORMICK LINEARISATION of q[t] = RA[t] * visits[t]
    # # 0 <= RA[t] <= S,  0 <= visits[t] <= N
    # for t in technicians:
    #     S = total_straws_available
    #     m.addConstr(q[t] <= S * visits[t], name=f"McC_ub1_{t}")
    #     m.addConstr(q[t] <= N * RA[t], name=f"McC_ub2_{t}")
    #     m.addConstr(q[t] >= S * visits[t] + N * RA[t] - S * N, name=f"McC_lb_{t}")


    # # 6. CAPACITY LIMIT (uses linearised product q[t])
    # for t in technicians:
    #     m.addConstr(
    #         gp.quicksum(
    #             x[i, f, t]
    #             for i in all_nodes
    #             for f in farmers
    #             if i != f
    #         )
    #         <=
    #         initial_straws[t] + q[t],
    #         name=f"CapLimit_{t}"
    #     )

    # # 7. MTZ SUBTOUR ELIMINATION [NOT NEEDED, I THINK THE STRAW COUNT SHOULD ACT AS A MTZ CONSTRAINT ]
    # for t in technicians:
    #     origin = origin_of[t]
    #     m.addConstr(
    #         u[origin, t] == 0
    #     )
    #     for i in all_nodes:
    #         for j in all_nodes:
    #             if i == j:
    #                 continue
    #             if i != origin and j != origin:
    #                 m.addConstr(
    #                     u[i, t]
    #                     - u[j, t]
    #                     + (N - 1) * x[i, j, t]
    #                     <= N - 2
    #                 )

    # # 8. DON'T ENTER OTHER TECHNICIANS' HOME NODES [IT SHOULD BE SUB OPTIMAL]
    # tech_origin_nodes = set(origin_of.values())
    # for t in technicians:
    #     my_origin = origin_of[t]
    #     for tech_node in tech_origin_nodes:
    #         if tech_node == my_origin:
    #             continue
    #         m.addConstr(
    #             gp.quicksum(
    #                 x[i, tech_node, t]
    #                 for i in all_nodes
    #                 if i != tech_node
    #             ) == 0
    #         )
    #         m.addConstr(
    #             gp.quicksum(
    #                 x[tech_node, j, t]
    #                 for j in all_nodes
    #                 if j != tech_node
    #             ) == 0
    #         )


    m.Params.LazyConstraints = 1

    def subtour_callback(model, where):
        if where != GRB.Callback.MIPSOL:
            return

        x_sol = {k: model.cbGetSolution(v) for k, v in x.items()}

        for t in technicians:
            origin = origin_of[t]
            succ = {
                i: j
                for (i, j, tt), val in x_sol.items()
                if tt == t and val > 0.5 and i != j
            }
            for subtour in _find_subtours(succ, all_nodes, origin):
                model.cbLazy(
                    gp.quicksum(
                        x[i, j, t]
                        for i in subtour
                        for j in subtour
                        if i != j
                    ) <= len(subtour) - 1
                )

    m.optimize(subtour_callback)
    m.write(os.path.join("logs", "model.lp"))
    routes = {}
    arcs   = {t: [] for t in technicians}

    if m.SolCount > 0:

        print("\nOptimal refill amounts per technician:")
        for t in technicians:
            print(f"  {t}: RA = {int(round(RA[t].X))}")

        x_vals = m.getAttr("X", x)

        # Collect all active arcs per technician
        for (i, j, tt), val in x_vals.items():
            if val > 0.5 and i != j:
                arcs[tt].append((i, j))

        for t in technicians:

            origin = origin_of[t]

            succ = {}

            for (i, j, tt), val in x_vals.items():

                if tt == t and val > 0.5 and i != j:
                    succ[i] = j

            path = [origin]

            current = origin
            seen = {origin}

            while current in succ:

                nxt = succ[current]

                if nxt in seen:
                    if nxt == origin:
                        path.append(nxt)  # close the route back to home
                    break

                path.append(nxt)

                seen.add(nxt)
                current = nxt

            routes[t] = path

    return routes, arcs