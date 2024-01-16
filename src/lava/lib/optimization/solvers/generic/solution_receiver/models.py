# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import numpy as np
from lava.lib.optimization.solvers.generic.monitoring_processes\
    .solution_readout.process import SolutionReadout
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import (
    PyLoihiProcessModel,
    PyAsyncProcessModel
)
from bitstring import Bits
from lava.magma.core.model.py.ports import PyInPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.async_protocol import AsyncProtocol

from lava.lib.optimization.solvers.generic.solution_receiver.process import \
    (
    SolutionReceiver, SpikeIntegrator
)
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.proc.sparse.process import Sparse, DelaySparse
from lava.utils.weightutils import SignMode
from lava.proc import embedded_io as eio
from scipy.sparse import csr_matrix


@implements(SolutionReceiver, protocol=AsyncProtocol)
@requires(CPU)
class SolutionReceiverPyModel(PyAsyncProcessModel):
    """CPU model for the SolutionReadout process.
    The process receives two types of messages, an updated cost and the
    state of
    the solver network representing the current candidate solution to an
    OptimizationProblem. Additionally, a target cost can be defined by the
    user, once this cost is reached by the solver network, this process
    will request the runtime service to pause execution.
    """

    best_state: np.ndarray = LavaPyType(np.ndarray, np.int8, 32)
    best_timestep: np.ndarray = LavaPyType(np.ndarray, np.int32, 32)
    best_cost: np.ndarray = LavaPyType(np.ndarray, np.int32, 32)
    num_message_bits: np.ndarray = LavaPyType(np.ndarray, np.int8, 32)

    results_in: PyInPort = LavaPyType(
        PyInPort.VEC_DENSE, np.int32, precision=32
    )

    def run_async(self):
        num_message_bits = self.num_message_bits[0]
        num_vars = self.best_state.shape[0]

        print("+" * 20)
        print(self.best_state)
        print("+" * 20)
        results_buffer = np.zeros(self.results_in._shape)
        print("Starting reception")
        while self._check_if_input(results_buffer):
            print("In while loop!")
            results_buffer = self.results_in.recv()
        print("Finished while loop")
        print(f"{results_buffer=}")
        self.best_cost, self.best_timestep, _ = self._decompress_state(
            compressed_states=results_buffer,
            num_message_bits=num_message_bits,
            num_vars=num_vars)
        print(f"{self.best_cost=}")
        print(f"{self.best_timestep=}")
        print("-" * 20)

        # best states are returned with a delay of 1 timestep
        results_buffer = self.results_in.recv()
        print("Received further results")
        _, _, states = self._decompress_state(
            compressed_states=results_buffer,
            num_message_bits=num_message_bits,
            num_vars=num_vars) #[:self.best_state.shape[0]]
        print("Finished")
        print(f"{self.best_state=}")
        print(f"{states=}")
        self.best_state = states
        print(f"{self.best_state}")
        self._req_pause = True

    @staticmethod
    def _check_if_input(results_buffer):
        return not results_buffer[1] > 0

    @staticmethod
    def _decompress_state(compressed_states, num_message_bits, num_vars):
        """Add info!"""
        cost = int(compressed_states[0])
        timestep = int(compressed_states[1])
        states = (compressed_states[2:, None] & (
                1 << np.arange(0, num_message_bits))) != 0
                #1 << np.arange(num_message_bits - 1, -1, -1))) != 0
        # reshape into a 1D array
        states.reshape(-1)
        # If n_vars is not a multiple of num_message_bits, then last entries
        # must be cut off
        states = states.astype(np.int8).flatten()[:num_vars]
        return cost, timestep, states




"""
def test_code():

    # Assuming you have a 32-bit integer numpy array
    original_array = np.array([4294967295, 2147483647, 0, 8983218],
                              dtype=np.uint32)

    # Use bitwise AND operation to convert each integer to a boolean array
    boolean_array = (original_array[:, None] & (1 << np.arange(31, -1, -1))) != 0

    # Display the result
    print(boolean_array)
"""

@implements(proc=SolutionReadout, protocol=LoihiProtocol)
@requires(CPU)
class SolutionReadoutModel(AbstractSubProcessModel):
    """Model for the SolutionReadout process.

    """

    def __init__(self, proc):
        num_message_bits = proc.proc_params.get("num_message_bits")

        # Define the dense input layer
        num_bin_variables = proc.proc_params.get("num_bin_variables")
        num_spike_integrators = proc.proc_params.get("num_spike_integrators")

        connection_config = proc.proc_params.get("connection_config")

        weights_state_in_0 = self._get_input_weights(
            num_vars=num_bin_variables,
            num_spike_int=num_spike_integrators,
            num_vars_per_int=num_message_bits,
            weight_exp=0
        )
        self.synapses_state_in_0 = Sparse(
            weights=weights_state_in_0,
            #sign_mode=SignMode.EXCITATORY,
            num_weight_bits=8,
            num_message_bits=num_message_bits,
            weight_exp=0,
        )

        weights_state_in_1 = self._get_input_weights(
            num_vars=num_bin_variables,
            num_spike_int=num_spike_integrators,
            num_vars_per_int=num_message_bits,
            weight_exp=8
        )
        self.synapses_state_in_1 = Sparse(
            weights=weights_state_in_1,
            #sign_mode=SignMode.EXCITATORY,
            num_weight_bits=8,
            num_message_bits=num_message_bits,
            weight_exp=8,
        )

        weights_state_in_2 = self._get_input_weights(
            num_vars=num_bin_variables,
            num_spike_int=num_spike_integrators,
            num_vars_per_int=num_message_bits,
            weight_exp=16
        )
        self.synapses_state_in_2 = Sparse(
            weights=weights_state_in_2,
            #sign_mode=SignMode.EXCITATORY,
            num_weight_bits=8,
            num_message_bits=num_message_bits,
            weight_exp=16,
        )

        weights_state_in_3 = self._get_input_weights(
            num_vars=num_bin_variables,
            num_spike_int=num_spike_integrators,
            num_vars_per_int=num_message_bits,
            weight_exp=24
        )
        self.synapses_state_in_3 = Sparse(
            weights=weights_state_in_3,
            #sign_mode=SignMode.EXCITATORY,
            num_weight_bits=8,
            num_message_bits=num_message_bits,
            weight_exp=24,
        )

        weights_cost_in = self._get_cost_in_weights(
            num_spike_int=num_spike_integrators,
        )
        #print("weights_cost_in", weights_cost_in)
        self.synapses_cost_in = Sparse(
            weights=weights_cost_in,
            num_weight_bits=8,
            num_message_bits=32,
        )

        weights_timestep_in = self._get_timestep_in_weights(
            num_spike_int=num_spike_integrators,
        )
        #print("weights_timestep_in", weights_timestep_in)
        self.synapses_timestep_in = Sparse(
            weights=weights_timestep_in,
            #sign_mode=SignMode.EXCITATORY,
            num_weight_bits=8,
            num_message_bits=32,
        )

        self.spike_integrators = SpikeIntegrator(shape=(num_spike_integrators,))

        self.solution_receiver = SolutionReceiver(
            shape=(1,),
            num_variables = num_bin_variables,
            num_spike_integrators = num_spike_integrators,
            num_message_bits = num_message_bits,
            best_cost_init = proc.best_cost.get(),
            best_state_init = proc.best_state.get(),
            best_timestep_init = proc.best_timestep.get()
        )

        # Connect the parent InPort to the InPort of the child-Process.
        proc.in_ports.states_in.connect(self.synapses_state_in_0.s_in)
        proc.in_ports.states_in.connect(self.synapses_state_in_1.s_in)
        proc.in_ports.states_in.connect(self.synapses_state_in_2.s_in)
        proc.in_ports.states_in.connect(self.synapses_state_in_3.s_in)
        proc.in_ports.cost_in.connect(self.synapses_cost_in.s_in)
        proc.in_ports.timestep_in.connect(self.synapses_timestep_in.s_in)

        # Connect intermediate ports
        self.synapses_state_in_0.a_out.connect(self.spike_integrators.a_in)
        self.synapses_state_in_1.a_out.connect(self.spike_integrators.a_in)
        self.synapses_state_in_2.a_out.connect(self.spike_integrators.a_in)
        self.synapses_state_in_3.a_out.connect(self.spike_integrators.a_in)
        self.synapses_cost_in.a_out.connect(self.spike_integrators.a_in)
        self.synapses_timestep_in.a_out.connect(self.spike_integrators.a_in)

        self.spike_integrators.s_out.connect(
            self.solution_receiver.results_in, connection_config)

        # Create aliases for variables
        proc.vars.best_state.alias(self.solution_receiver.best_state)
        proc.vars.best_timestep.alias(self.solution_receiver.best_timestep)
        proc.vars.best_cost.alias(self.solution_receiver.best_cost)

    @staticmethod
    def _get_input_weights(num_vars, num_spike_int, num_vars_per_int, weight_exp):
        """To be verified. Deprecated due to efficiency"""

        weights = np.zeros((num_spike_int, num_vars), dtype=np.uint8)
        #print(f"{num_vars=}")
        #print(f"{num_spike_int=}")
        #print(f"{num_vars_per_int=}")
        # The first two SpikeIntegrators receive best_cost and best_timestep
        for spike_integrator in range(2, num_spike_int - 1):
            variable_start = num_vars_per_int * (spike_integrator - 2) + weight_exp
            weights[spike_integrator, variable_start:variable_start +
                                                     8] = np.power(2,
                                                                   np.arange(8))
        # The last spike integrator might be connected by less than
        # num_vars_per_int neurons
        # This happens when mod(num_variables, num_vars_per_int) != 0
        variable_start = num_vars_per_int * (num_spike_int - 3) + weight_exp
        weights[-1, variable_start:] = np.power(2, np.arange(weights.shape[1]-variable_start))

        #print("=" * 20)
        #print(f"{weights=}")
        #print("=" * 20)

        return csr_matrix(weights)

    @staticmethod
    def _get_state_in_weights_index(num_vars, num_spike_int, num_vars_per_int):
        """To be verified"""
        weights = np.zeros((num_spike_int, num_vars), dtype=np.int8)

        # Compute the indices for setting the values to 1
        indices = np.arange(0, num_vars_per_int * (num_spike_int - 1), num_vars_per_int)

        # Set the values to 1 using array indexing
        weights[:num_spike_int-1, indices:indices + num_vars_per_int] = 1

        # Set the values for the last spike integrator
        weights[-1, num_vars_per_int * (num_spike_int - 1):num_vars] = 1

        return weights

    @staticmethod
    def _get_cost_in_weights(num_spike_int: int) -> csr_matrix:
        weights = np.zeros((num_spike_int, 1), dtype=int)
        weights[0,0] = 1
        return csr_matrix(weights)
        
    @staticmethod
    def _get_timestep_in_weights(num_spike_int: int) -> csr_matrix:
        weights = np.zeros((num_spike_int, 1), dtype=int)
        weights[1,0] = 1
        return csr_matrix(weights)
