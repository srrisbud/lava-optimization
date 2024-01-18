# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import pprint
import unittest

import numpy as np
import networkx as ntx
import matplotlib.pyplot as plt

from lava.lib.optimization.apps.scheduler.problems import (
    SatelliteScheduleProblem)
from lava.lib.optimization.apps.scheduler.resource_estimator import (
    ResourceEstimatorSatScheduler)
from lava.lib.optimization.apps.scheduler.solver import SatelliteScheduler

from lava.magma.compiler.subcompilers.nc.ncproc_compiler import CompilerOptions


class TestSatelliteSchedulingProblem(unittest.TestCase):

    def setUp(self) -> None:
        n_sat = 250
        n_req = 10000
        reqs_per_sat = np.floor(n_req / n_sat).astype(int)
        requests = np.zeros((n_req, 2))
        delta = 1 / n_sat
        y_sats = (np.arange(1, n_sat + 1) - (1 / 2)) * delta
        np.random.seed(42)
        for j in range(n_sat):
            s_idx = reqs_per_sat * j
            e_idx = s_idx + reqs_per_sat
            requests[s_idx:e_idx, :] = np.random.random(size=(reqs_per_sat, 2))
            requests[s_idx:e_idx, 1] = (
                    (delta * requests[s_idx:e_idx,
                                      1]) + (y_sats[j] - (delta / 2)))
        self.ssp = SatelliteScheduleProblem(num_satellites=n_sat,
                                            num_requests=n_req,
                                            requests=requests,
                                            view_height=1.5,
                                            turn_rate=2,
                                            seed=42)
        self.res_est = (
            ResourceEstimatorSatScheduler(self.ssp,
                                          num_wgt_bits=20,
                                          neurons_per_core=100,
                                          max_neurons_per_core=4095,
                                          max_axon_mem_to_syn_mem_ratio=0.1))

        self.schr = SatelliteScheduler(ssp=self.ssp,
                                       qubo_weights=(4, 20),
                                       )
        self.schr.lava_backend = "Loihi2"
        self.schr.qubo_hyperparams = ({"temperature": 1}, True)

    def test_init(self):
        self.assertIsInstance(self.ssp, SatelliteScheduleProblem)

    def test_generate(self):
        np.set_printoptions(linewidth=10000, threshold=100000)
        self.res_est.partition()
        all = np.hstack((self.res_est.num_syn_map_enries_per_core,
                         self.res_est.num_syn_mem_words_per_core,
                         self.res_est.num_axon_map_entries_per_core,
                         self.res_est.num_axon_mem_words_per_core))
        print(all)
        # print(self.res_est.num_input_axons_per_core)
        # print(self.res_est.num_output_axons_per_core)
        # self.res_est.problem.plot_problem()
        # CompilerOptions.verbose = True
        # CompilerOptions.show_resource_count = True
        # self.schr.solve_with_lava_qubo(timeout=100)


if __name__ == '__main__':
    unittest.main()
