import hexaly.optimizer


def solve_technician_routing_hexaly(
    all_nodes,
    farmers,
    technicians,
    office_node,
    initial_straws,
    total_straws_available,
    full_distance,
    origin_of,
    time_limit=None,
    verbose=False,
):

    all_nodes = sorted(set(all_nodes))
    farmers = sorted(set(farmers))
    technicians = list(technicians)

    nb_farmers = len(farmers)
    nb_technicians = len(technicians)

    index_map = {node: idx for idx, node in enumerate(all_nodes)}

    def dist(i, j):
        if isinstance(full_distance, dict):
            return full_distance[i][j]
        return full_distance[index_map[i]][index_map[j]]

    with hexaly.optimizer.HexalyOptimizer() as optimizer:

        optimizer.param.verbosity = 2 if verbose else 0
        if time_limit is not None:
            optimizer.param.time_limit = time_limit

        model = optimizer.model

        # ─────────────────────────────
        # DECISION VARIABLES
        # ─────────────────────────────
        routes = [model.list(nb_farmers) for _ in range(nb_technicians)]
        model.constraint(model.partition(*routes))

        RA = [model.int(0, total_straws_available) for _ in range(nb_technicians)]

        NO_OFFICE = nb_farmers + 1
        office_pos = [model.int(0, NO_OFFICE) for _ in range(nb_technicians)]

        model.constraint(model.sum(RA) <= total_straws_available)

        tech_distances = []

        for t_idx, t in enumerate(technicians):

            origin = origin_of[t]
            route = routes[t_idx]
            ra = RA[t_idx]
            op = office_pos[t_idx]
            init_s = initial_straws[t]

            n = model.count(route)

            # capacity constraint
            model.constraint(init_s + ra >= n)

            # office needed iff refill required
            model.constraint(model.iif(n > init_s, op <= n, op == NO_OFFICE))

            # distances
            d_o2f = model.array([dist(origin, f) for f in farmers])
            d_f2o = model.array([dist(f, origin) for f in farmers])
            d_f2off = model.array([dist(f, office_node) for f in farmers])
            d_off2f = model.array([dist(office_node, f) for f in farmers])
            d_ff = model.array([[dist(f1, f2) for f2 in farmers] for f1 in farmers])

            d_off_o = dist(office_node, origin)
            d_o_off = dist(origin, office_node)

            # base
            internal = model.sum(
                route,
                model.lambda_function(
                    lambda i: model.iif(
                        i == 0,
                        model.at(d_o2f, model.at(route, 0)),
                        model.at(
                            d_ff,
                            model.at(route, model.max(0, i - 1)),
                            model.at(route, i),
                        ),
                    )
                ),
            )

            back = model.iif(n > 0, model.at(d_f2o, model.at(route, n - 1)), 0)

            base_dist = model.iif(n > 0, internal + back, 0)

            # with office
            internal_off = model.sum(
                route,
                model.lambda_function(
                    lambda i: model.iif(
                        i == 0,
                        model.iif(
                            op == 0,
                            d_o_off + model.at(d_off2f, model.at(route, 0)),
                            model.at(d_o2f, model.at(route, 0)),
                        ),
                        model.iif(
                            i == op,
                            model.at(d_f2off, model.at(route, model.max(0, i - 1)))
                            + model.at(d_off2f, model.at(route, i)),
                            model.at(
                                d_ff,
                                model.at(route, model.max(0, i - 1)),
                                model.at(route, i),
                            ),
                        ),
                    )
                ),
            )

            back_off = model.iif(
                n > 0,
                model.iif(
                    op == n,
                    model.at(d_f2off, model.at(route, n - 1)) + d_off_o,
                    model.at(d_f2o, model.at(route, n - 1)),
                ),
                0,
            )

            tech_cost = model.iif(
                op == NO_OFFICE,
                base_dist,
                model.iif(n > 0, internal_off + back_off, 0),
            )

            tech_distances.append(tech_cost)

        model.minimize(model.sum(tech_distances))
        model.close()

        optimizer.solve()

        # ─────────────────────────────
        # EXTRACTION
        # ─────────────────────────────

        routes_out = {}
        arcs_out = {t: [] for t in technicians}
        RA_out = {}

        print("\nOptimal RA values:")
        for t_idx, t in enumerate(technicians):
            RA_out[t] = int(round(RA[t_idx].value))
            print(f"  {t}: RA = {RA_out[t]}")

        for t_idx, t in enumerate(technicians):

            origin = origin_of[t]
            route = routes[t_idx]
            op_val = int(round(office_pos[t_idx].value))

            seq = [farmers[i] for i in route.value]

            path = [origin]

            for i, f in enumerate(seq):
                if op_val != NO_OFFICE and i == op_val:
                    path.append(office_node)
                path.append(f)

            if op_val != NO_OFFICE and op_val == len(seq):
                path.append(office_node)

            path.append(origin)

            routes_out[t] = path

            for i in range(len(path) - 1):
                arcs_out[t].append((path[i], path[i + 1]))

        return routes_out, arcs_out, RA_out