import math
import numpy as np

from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol

from lava.lib.optimization.solvers.bayesian.solver import BayesianSolver


class AckleyFuncProcess(AbstractProcess):
    """Process defining the architecture of the Ackley function
    """
    def __init__(self, num_params: int = 2, num_objectives: int = 1,
                 **kwargs) -> None:
        """initialize the AckleyFuncProcess

        Parameters
        ----------
        num_params : int
            an integer specifying the number of parameters within the
            search space
        num_objectives : int
            an integer specifying the number of qualitative attributes
            used to measure the black-box function
        """
        super().__init__(**kwargs)

        # Internal State Variables
        self.num_params = Var((1,), init=num_params)
        self.num_objectives = Var((1,), init=num_objectives)

        # Input/Output Ports
        self.x_in = InPort((num_params, 1))
        self.y_out = OutPort(((num_params + num_objectives), 1))


@implements(proc=AckleyFuncProcess, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PyAckleyFuncProcessModel(PyLoihiProcessModel):
    """
    A Python-based implementation of the Ackley function process.
    """

    x_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float64)
    y_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.float64)

    num_params = LavaPyType(int, int)
    num_objectives = LavaPyType(int, int)

    def run_spk(self) -> None:
        """tick the model forward by one time-step"""
        print(f"time-step: {self.time_step}")
        x = self.x_in.recv()
        print(f"Received:\n{x}\n-----")
        y = -20 * math.exp(-0.2 * math.sqrt(0.5 * (x[0]**2 + x[1]**2)))
        y -= math.exp(0.5 * (math.cos(2 * math.pi * x[0]) + \
            math.cos(2 * math.pi * x[1])))
        y += math.e + 20

        output_length: int = self.num_params + self.num_objectives
        output = np.ndarray(shape=(output_length, 1))
        output[0][0], output[1][0], output[2][0] = x[0], x[1], y
        print(f"Output:\n{output}")
        self.y_out.send(output)


def main():

    search_space: np.ndarray = np.array([
        ["continuous", np.float64(-5), np.float64(5), np.nan, "x0"],
        ["categorical", np.nan, np.nan, np.arange(-2, 5, 0.125), "x1"]
    ], dtype=object)

    num_ips: int = 5
    seed: int = 0

    solver = BayesianSolver(
        acq_func_config={"type": "gp_hedge"},
        acq_opt_config={"type": "auto"},
        ip_gen_config={"type": "random"},
        num_ips=num_ips,
        seed=seed
    )

    # 1) initialize the Ackley function process
    problem = AckleyFuncProcess()

    # 2) specify the experiment name and the number of optimization iteration
    experiment_name: str = "bayesian_tutorial_results"
    num_iter: int = 100

    # 3) solve the problem!
    print("Solving the problem")
    solver.solve(
        name=experiment_name,
        num_iter=num_iter,
        problem=problem,
        search_space=search_space,
    )


if __name__ == "__main__":
    main()
