# planning for small test suite before reading pytest docs

# 1. x_0 = (0, 1), x_1 = (1, 1), x_2 = (1, 0)
# (a). gamma = 0: solution should be build all facilities: objective = 0.0 (for all solvers).
# (b). gamma = 100: solution should be build no facilities and serve all points with x_0:
# - objective (same for all solvers since "empty" facility set for all 3): (1) + (1) = 2
# - \sum_t max_min_service_cost(t)
