import unittest
import numpy as np
from pprint import pprint
from lava.lib.optimization.apps.vrp.problems import VRP
from lava.lib.optimization.apps.vrp.solver import VRPConfig, VRPSolver, \
    CoreSolver


class testVRPSolver(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(4231)
        # random_coords = [(np.random.randint(0, 26),
        #                   np.random.randint(0, 26)) for _ in range(8)]
        random_coords = [(13, 16), (9, 13), (14, 9), (14, 18),
                         (12, 14), (8, 11), (15, 10), (20, 8)]
        self.vehicle_coords = random_coords[:3]
        self.node_coords = random_coords[3:]
        self.edges = [(4, 5), (4, 6), (6, 8), (8, 7), (7, 6), (8, 5)]
        self.vrp_instance_natural_edges = VRP(
            node_coords=self.node_coords, vehicle_coords=self.vehicle_coords)
        self.vrp_instance_user_edges = VRP(
            node_coords=self.node_coords, vehicle_coords=self.vehicle_coords,
            edges=self.edges)

    def test_init(self):
        solver = VRPSolver(vrp=self.vrp_instance_natural_edges)
        self.assertIsInstance(solver, VRPSolver)

    def test_solve_vrpy(self):
        solver = VRPSolver(vrp=self.vrp_instance_user_edges)
        result = solver.solve()
        gold = {1: [3], 2: [1, 4, 5], 3: [2, 7, 6, 8]}
        self.assertDictEqual(gold, result)

    def test_solve_lava_qubo(self):
        solver = VRPSolver(vrp=self.vrp_instance_natural_edges)
        scfg = VRPConfig(backend="Loihi2", log_level=20,
                         core_solver=CoreSolver.LAVA_QUBO, timeout=10000)
        report = solver.solve(scfg=scfg)


if __name__ == '__main__':
    unittest.main()