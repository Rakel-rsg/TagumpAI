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
    """Solve the technician routing problem with Hexaly.

    Each technician's tour is a Hexaly list over farmer indices.  The partition
    constraint guarantees every farmer is served by exactly one technician.
    When a technician needs to refill (more farmers assigned than initial
    straws), the optimizer also chooses *where* in the tour to insert the
    office visit via the integer variable office_pos[t].

    Parameters mirror solve_technician_routing exactly so the two solvers are
    drop-in replacements for one another.
    """
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

        # ── Decision variables ─────────────────────────────────────────────
        # One list per technician; elements are farmer indices 0..nb_farmers-1.
        # The list captures the ordered sequence of farmer visits.
        routes = [model.list(nb_farmers) for _ in range(nb_technicians)]

        # Partition: every farmer index in exactly one route.
        model.constraint(model.partition(*routes))

        # Straw refill amount per technician (picked up at the one office visit).
        RA = [model.int(0, total_straws_available) for _ in range(nb_technicians)]

        # Office insertion point: position k means the office is visited between
        # the (k-1)-th and k-th farmer in the route (k=0 → office before any
        # farmer; k=count → office after the last farmer, just before going home).
        # Sentinel nb_farmers+1 encodes "no office visit needed".
        NO_OFFICE = nb_farmers + 1
        office_pos = [model.int(0, NO_OFFICE) for _ in range(nb_technicians)]

        # ── Global constraints ─────────────────────────────────────────────
        model.constraint(model.sum(RA) <= total_straws_available)

        # ── Per-technician cost and constraints ────────────────────────────
        tech_distances = []

        for t_idx, t in enumerate(technicians):
            origin = origin_of[t]
            route = routes[t_idx]
            ra = RA[t_idx]
            op = office_pos[t_idx]
            init_s = initial_straws[t]
            n = model.count(route)

            # Straw feasibility: must carry enough straws for every assigned farmer.
            model.constraint(init_s + ra >= n)

            # Office must be visited iff the technician needs a refill.
            # "needs refill" ↔ n > init_s  ↔  op != NO_OFFICE
            model.constraint(model.iif(n > init_s, op <= n, op == NO_OFFICE))

            # ── Distance arrays (Hexaly constants, indexed by farmer index) ──
            d_o2f   = model.array([dist(origin, f)      for f in farmers])
            d_f2o   = model.array([dist(f, origin)      for f in farmers])
            d_f2off = model.array([dist(f, office_node) for f in farmers])
            d_off2f = model.array([dist(office_node, f) for f in farmers])
            # 2-D farmer × farmer distance matrix.
            d_ff = model.array([[dist(f1, f2) for f2 in farmers] for f1 in farmers])

            dist_origin_office = dist(origin, office_node)
            dist_office_origin = dist(office_node, origin)

            # ── Base route distance (no office) ───────────────────────────
            # sum_{i=0}^{n-1}  dist(prev_node, route[i])
            # where prev_node is origin for i=0, else route[i-1].
            # model.lambda_function requires exactly 1 argument; variables are
            # captured via closure (Hexaly evaluates the lambda immediately at
            # model-construction time, so loop-variable aliasing is not an issue).
            # model.max(0, i-1) guards against a symbolic index of -1 when i==0;
            # that branch is never selected by the enclosing iif.
            internal = model.sum(
                route,
                model.lambda_function(
                    lambda i: model.iif(
                        i == 0,
                        model.at(d_o2f, model.at(route, 0)),
                        model.at(d_ff, model.at(route, model.max(0, i - 1)), model.at(route, i)),
                    )
                ),
            )
            back_to_origin = model.iif(
                n > 0, model.at(d_f2o, model.at(route, n - 1)), 0
            )
            base_dist = model.iif(n > 0, internal + back_to_origin, 0)

            # ── Office detour overhead ─────────────────────────────────────
            # The optimizer chooses op ∈ {0, …, n} to minimise the detour.
            # Overhead replaces the direct edge at insertion point op with
            # two edges routed through the office:
            #   op == 0  : origin → office → route[0]
            #              instead of  origin → route[0]
            #   0 < op < n : route[op-1] → office → route[op]
            #              instead of  route[op-1] → route[op]
            #   op == n  : route[n-1] → office → origin
            #              instead of  route[n-1] → origin
            # When op == NO_OFFICE no overhead is added.

            # Overhead in the lambda sum: replace the incoming edge at each
            # position when that position equals op.
            internal_with_office = model.sum(
                route,
                model.lambda_function(
                    lambda i: model.iif(
                        i == 0,
                        model.iif(
                            op == 0,
                            # office inserted before first farmer
                            dist_origin_office + model.at(d_off2f, model.at(route, 0)),
                            # no office at this edge
                            model.at(d_o2f, model.at(route, 0)),
                        ),
                        model.iif(
                            i == op,
                            # office inserted between route[i-1] and route[i]
                            model.at(d_f2off, model.at(route, model.max(0, i - 1)))
                            + model.at(d_off2f, model.at(route, i)),
                            # direct edge
                            model.at(d_ff, model.at(route, model.max(0, i - 1)), model.at(route, i)),
                        ),
                    )
                ),
            )
            # Last edge: route[n-1] → origin (or → office → origin when op == n).
            back_with_office = model.iif(
                n > 0,
                model.iif(
                    op == n,
                    model.at(d_f2off, model.at(route, n - 1)) + dist_office_origin,
                    model.at(d_f2o, model.at(route, n - 1)),
                ),
                0,
            )

            # Choose full expression based on whether office is visited.
            tech_dist = model.iif(
                op == NO_OFFICE,
                base_dist,
                model.iif(n > 0, internal_with_office + back_with_office, 0),
            )
            tech_distances.append(tech_dist)

        model.minimize(model.sum(tech_distances))
        model.close()
        optimizer.solve()

        # ── Extract solution ───────────────────────────────────────────────
        routes_out = {}
        arcs_out = {t: [] for t in technicians}

        print("\nOptimal refill amounts per technician:")
        for t_idx, t in enumerate(technicians):
            print(f"  {t}: RA = {int(round(RA[t_idx].value))}")

        for t_idx, t in enumerate(technicians):
            origin = origin_of[t]
            route = routes[t_idx]
            op_val = int(round(office_pos[t_idx].value))
            count_val = route.value.count()

            farmer_seq = [farmers[route.value[i]] for i in range(count_val)]

            # Reconstruct path inserting office at op_val (if not sentinel).
            path = [origin]
            for i, f in enumerate(farmer_seq):
                if op_val != NO_OFFICE and i == op_val:
                    path.append(office_node)
                path.append(f)
            # Office after last farmer (op_val == count_val).
            if op_val != NO_OFFICE and op_val == count_val:
                path.append(office_node)
            path.append(origin)

            routes_out[t] = path
            for i in range(len(path) - 1):
                arcs_out[t].append((path[i], path[i + 1]))

        return routes_out, arcs_out
