# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
from lava.lib.optimization.problems.problems import SparseCodingLASSO


class LCASolver:
    """Solver for LASSO problems arising in sparse coding, using locally
    competitive algorithm"""

    def __init__(self):
        pass

    def solve(self, problem: SparseCodingLASSO, num_iter: int = 128):
        pass

