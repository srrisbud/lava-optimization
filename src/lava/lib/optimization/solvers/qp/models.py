# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

"""
Implement behaviors (models) of the processes defined in processes.py
For further documentation please refer to processes.py
"""
import numpy as np
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.lib.optimization.solvers.qp.processes import (
    ConstraintDirections,
    ConstraintCheck,
    ConstraintNeurons,
    ConstraintNormals,
    QuadraticConnectivity,
    SolutionNeurons,
    GradientDynamics,
    SigmaDeltaSolutionNeurons,
    SigmaNeurons,
)


@implements(proc=ConstraintDirections, protocol=LoihiProtocol)
@requires(CPU)
class PyCDModel(PyLoihiProcessModel):
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    a_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    weights: np.ndarray = LavaPyType(np.ndarray, float)
    # Profiling
    synops: int = LavaPyType(int, np.int32)
    neurops: int = LavaPyType(int, np.int32)
    spikeops: int = LavaPyType(int, np.int32)

    def run_spk(self):
        s_in = self.s_in.recv()
        # process behavior: matrix multiplication
        self.synops += np.count_nonzero(self.weights[:, s_in.nonzero()])
        a_out = self.weights @ s_in
        self.a_out.send(a_out)


@implements(proc=SigmaNeurons, protocol=LoihiProtocol)
@requires(CPU)
class PySigNeurModel(PyLoihiProcessModel):
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float64)
    a_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.float64)
    x_internal: np.ndarray = LavaPyType(np.ndarray, np.float64)
    # Profiling
    synops: int = LavaPyType(int, np.int32)
    neurops: int = LavaPyType(int, np.int32)
    spikeops: int = LavaPyType(int, np.int32)

    def run_spk(self):
        s_in = self.s_in.recv()
        # process behavior: constraint violation check
        self.x_internal += s_in
        a_out = self.x_internal
        self.neurops += np.count_nonzero(a_out)
        self.spikeops += np.count_nonzero(a_out)
        self.a_out.send(a_out)


@implements(proc=ConstraintNeurons, protocol=LoihiProtocol)
@requires(CPU)
class PyCNeuModel(PyLoihiProcessModel):
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float64)
    a_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.float64)
    thresholds: np.ndarray = LavaPyType(np.ndarray, np.float64)
    # Profiling
    synops: int = LavaPyType(int, np.int32)
    neurops: int = LavaPyType(int, np.int32)
    spikeops: int = LavaPyType(int, np.int32)

    def run_spk(self):
        s_in = self.s_in.recv()
        # process behavior: constraint violation check
        a_out = (s_in - self.thresholds) * (s_in > self.thresholds)
        self.neurops += np.count_nonzero(a_out)
        self.spikeops += np.count_nonzero(a_out)
        self.a_out.send(a_out)


@implements(proc=QuadraticConnectivity, protocol=LoihiProtocol)
@requires(CPU)
class PyQCModel(PyLoihiProcessModel):
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float64)
    a_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.float64)
    weights: np.ndarray = LavaPyType(np.ndarray, np.float64)
    # Profiling
    synops: int = LavaPyType(int, np.int32)
    neurops: int = LavaPyType(int, np.int32)
    spikeops: int = LavaPyType(int, np.int32)

    def run_spk(self):
        s_in = self.s_in.recv()
        # process behavior: matrix multiplication
        self.synops += np.count_nonzero(self.weights[:, s_in.nonzero()])
        a_out = self.weights @ s_in
        self.a_out.send(a_out)


@implements(proc=SolutionNeurons, protocol=LoihiProtocol)
@requires(CPU)
class PySNModel(PyLoihiProcessModel):
    s_in_qc: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float64)
    a_out_qc: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.float64)
    s_in_cn: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float64)
    a_out_cc: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.float64)
    qp_neuron_state: np.ndarray = LavaPyType(np.ndarray, np.float64)
    grad_bias: np.ndarray = LavaPyType(np.ndarray, np.float64)
    alpha: np.ndarray = LavaPyType(np.ndarray, np.float64)
    beta: np.ndarray = LavaPyType(np.ndarray, np.float64)
    alpha_decay_schedule: int = LavaPyType(int, np.int32)
    beta_growth_schedule: int = LavaPyType(int, np.int32)
    decay_counter: int = LavaPyType(int, np.int32)
    growth_counter: int = LavaPyType(int, np.int32)

    # Profiling
    synops: int = LavaPyType(int, np.int32)
    neurops: int = LavaPyType(int, np.int32)
    spikeops: int = LavaPyType(int, np.int32)

    def run_spk(self):
        a_out = self.qp_neuron_state
        self.spikeops += np.count_nonzero(a_out)
        self.a_out_cc.send(a_out)
        self.a_out_qc.send(a_out)

        s_in_qc = self.s_in_qc.recv()
        s_in_cn = self.s_in_cn.recv()

        self.decay_counter += 1
        if self.decay_counter == self.alpha_decay_schedule:
            # TODO: guard against shift overflows in fixed-point
            self.alpha /= 2  # equivalent to right shift operation
            self.decay_counter = 0

        self.growth_counter += 1
        if self.growth_counter == self.beta_growth_schedule:
            # TODO: guard against shift overflows in fixed-point
            self.beta *= 2  # equivalent to left shift operation
            self.growth_counter = 0

        # process behavior: gradient update
        curr_state = (
            -self.alpha * (s_in_qc + self.grad_bias) - self.beta * s_in_cn
        )
        self.qp_neuron_state += curr_state
        self.neurops += np.count_nonzero(curr_state)


@implements(proc=SigmaDeltaSolutionNeurons, protocol=LoihiProtocol)
@requires(CPU)
class PySDSNModel(PyLoihiProcessModel):
    s_in_qc: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float64)
    a_out_qc: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.float64)
    s_in_cn: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float64)
    a_out_cc: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.float64)
    qp_neuron_state: np.ndarray = LavaPyType(np.ndarray, np.float64)
    prev_qp_neuron_state: np.ndarray = LavaPyType(np.ndarray, np.float64)
    grad_bias: np.ndarray = LavaPyType(np.ndarray, np.float64)
    theta: np.ndarray = LavaPyType(np.ndarray, np.float64)
    alpha: np.ndarray = LavaPyType(np.ndarray, np.float64)
    beta: np.ndarray = LavaPyType(np.ndarray, np.float64)
    theta_decay_schedule: int = LavaPyType(int, np.int32)
    alpha_decay_schedule: int = LavaPyType(int, np.int32)
    beta_growth_schedule: int = LavaPyType(int, np.int32)
    decay_counter_theta: int = LavaPyType(int, np.int32)
    decay_counter: int = LavaPyType(int, np.int32)
    growth_counter: int = LavaPyType(int, np.int32)

    # Profiling
    synops: int = LavaPyType(int, np.int32)
    neurops: int = LavaPyType(int, np.int32)
    spikeops: int = LavaPyType(int, np.int32)

    def run_spk(self):
        # Delta Operation
        delta_state = self.qp_neuron_state - self.prev_qp_neuron_state
        a_out_cc = delta_state * (np.abs(delta_state) >= self.theta)
        a_out_qc = self.qp_neuron_state * (np.abs(delta_state) >= self.theta)
        self.spikeops += np.count_nonzero(a_out_cc)
        self.a_out_cc.send(a_out_cc)
        self.a_out_qc.send(a_out_qc)

        s_in_qc = self.s_in_qc.recv()
        s_in_cn = self.s_in_cn.recv()

        self.decay_counter_theta += 1
        if self.decay_counter_theta == self.theta_decay_schedule:
            # TODO: guard against shift overflows in fixed-point
            self.theta /= 2  # equivalent to right shift operation
            self.decay_counter_theta = 0

        self.decay_counter += 1
        if self.decay_counter == self.alpha_decay_schedule:
            # TODO: guard against shift overflows in fixed-point
            self.alpha /= 2  # equivalent to right shift operation
            self.decay_counter = 0

        self.growth_counter += 1
        if self.growth_counter == self.beta_growth_schedule:
            # TODO: guard against shift overflows in fixed-point
            self.beta *= 2  # equivalent to left shift operation
            self.growth_counter = 0

        # process behavior: gradient update
        self.prev_qp_neuron_state = self.qp_neuron_state.copy()
        state_update = (
            -self.alpha * (s_in_qc + self.grad_bias) - self.beta * s_in_cn
        )
        self.qp_neuron_state += state_update * (
            np.abs(state_update) >= self.theta
        )
        self.neurops += np.count_nonzero(state_update)


@implements(proc=ConstraintNormals, protocol=LoihiProtocol)
@requires(CPU)
class PyCNorModel(PyLoihiProcessModel):
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float64)
    a_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.float64)
    weights: np.ndarray = LavaPyType(np.ndarray, np.float64)

    # Profiling
    synops: int = LavaPyType(int, np.int32)
    neurops: int = LavaPyType(int, np.int32)
    spikeops: int = LavaPyType(int, np.int32)

    def run_spk(self):
        s_in = self.s_in.recv()
        # process behavior: matrix multiplication
        self.synops += np.count_nonzero(self.weights[:, s_in.nonzero()])
        a_out = self.weights @ s_in
        self.a_out.send(a_out)


@implements(proc=ConstraintCheck, protocol=LoihiProtocol)
class SubCCModel(AbstractSubProcessModel):
    """Implement constraintCheckProcess behavior via sub Processes."""

    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float64)
    constraint_matrix: np.ndarray = LavaPyType(np.ndarray, np.float64)
    constraint_bias: np.ndarray = LavaPyType(np.ndarray, np.float64)
    x_internal: np.ndarray = LavaPyType(np.ndarray, np.float64)
    a_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.float64)

    # profiling
    cNeur_synops: int = LavaPyType(int, np.int32)
    cNeur_neurops: int = LavaPyType(int, np.int32)
    cNeur_spikeops: int = LavaPyType(int, np.int32)

    cD_synops: int = LavaPyType(int, np.int32)
    cD_neurops: int = LavaPyType(int, np.int32)
    cD_spikeops: int = LavaPyType(int, np.int32)

    def __init__(self, proc):
        """Builds sub Process structure of the Process."""
        constraint_matrix = proc.init_args.get("constraint_matrix", 0)
        constraint_bias = proc.init_args.get("constraint_bias", 0)
        x_int_init = proc.init_args.get("x_int_init", 0)
        sparse = proc.init_args.get("sparse", False)

        # Initialize subprocesses
        self.constraintDirections = ConstraintDirections(
            shape=constraint_matrix.shape,
            constraint_directions=constraint_matrix,
        )
        self.constraintNeurons = ConstraintNeurons(
            shape=constraint_bias.shape, thresholds=constraint_bias
        )

        if sparse:
            print("[INFO]: Using additional Sigma layer")
            self.sigmaNeurons = SigmaNeurons(
                shape=(constraint_matrix.shape[1], 1), x_int_init=x_int_init
            )

            proc.vars.x_internal.alias(self.sigmaNeurons.vars.x_internal)
            # connect subprocesses to obtain required process behavior
            proc.in_ports.s_in.connect(self.sigmaNeurons.in_ports.s_in)
            self.sigmaNeurons.out_ports.a_out.connect(
                self.constraintDirections.in_ports.s_in
            )

        else:
            proc.in_ports.s_in.connect(self.constraintDirections.in_ports.s_in)

        # remaining procesess to connect irrespective of sparsity
        self.constraintDirections.out_ports.a_out.connect(
            self.constraintNeurons.in_ports.s_in
        )
        self.constraintNeurons.out_ports.a_out.connect(proc.out_ports.a_out)

        # alias process variables to subprocess variables
        proc.vars.constraint_matrix.alias(
            self.constraintDirections.vars.weights
        )
        proc.vars.constraint_bias.alias(self.constraintNeurons.vars.thresholds)

        # profiling
        proc.vars.cNeur_synops.alias(self.constraintNeurons.vars.synops)
        proc.vars.cNeur_neurops.alias(self.constraintNeurons.vars.neurops)
        proc.vars.cNeur_spikeops.alias(self.constraintNeurons.vars.spikeops)
        proc.vars.cD_synops.alias(self.constraintDirections.vars.synops)
        proc.vars.cD_neurops.alias(self.constraintDirections.vars.neurops)
        proc.vars.cD_spikeops.alias(self.constraintDirections.vars.spikeops)


@implements(proc=GradientDynamics, protocol=LoihiProtocol)
class SubGDModel(AbstractSubProcessModel):
    """Implement gradientDynamics Process behavior via sub Processes."""

    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float64)
    hessian: np.ndarray = LavaPyType(np.ndarray, np.float64)
    constraint_matrix_T: np.ndarray = LavaPyType(
        np.ndarray,
        np.float64,
    )
    grad_bias: np.ndarray = LavaPyType(np.ndarray, np.float64)
    qp_neuron_state: np.ndarray = LavaPyType(np.ndarray, np.float64)
    alpha: np.ndarray = LavaPyType(np.ndarray, np.float64)
    beta: np.ndarray = LavaPyType(np.ndarray, np.float64)
    alpha_decay_schedule: int = LavaPyType(int, np.int32)
    beta_growth_schedule: int = LavaPyType(int, np.int32)
    a_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.float64)

    # profiling
    cN_synops: int = LavaPyType(int, np.int32)
    cN_neurops: int = LavaPyType(int, np.int32)
    cN_spikeops: int = LavaPyType(int, np.int32)

    qC_synops: int = LavaPyType(int, np.int32)
    qC_neurops: int = LavaPyType(int, np.int32)
    qC_spikeops: int = LavaPyType(int, np.int32)

    sN_synops: int = LavaPyType(int, np.int32)
    sN_neurops: int = LavaPyType(int, np.int32)
    sN_spikeops: int = LavaPyType(int, np.int32)

    def __init__(self, proc):
        """Builds sub Process structure of the Process."""
        hessian = proc.init_args.get("hessian", 0)
        shape_hess = hessian.shape
        shape_sol = (shape_hess[0], 1)
        constraint_matrix_T = proc.init_args.get("constraint_matrix_T", 0)
        shape_constraint_matrix_T = constraint_matrix_T.shape
        grad_bias = proc.init_args.get("grad_bias", np.zeros(shape_sol))
        qp_neuron_i = proc.init_args.get(
            "qp_neurons_init", np.zeros(shape_sol)
        )
        sparse = proc.init_args.get("sparse", False)
        theta = proc.init_args.get("theta", np.zeros(shape_sol))
        alpha = proc.init_args.get("alpha", np.ones(shape_sol))
        beta = proc.init_args.get("beta", np.ones(shape_sol))
        t_d = proc.init_args.get("theta_decay_schedule", 10000)
        a_d = proc.init_args.get("alpha_decay_schedule", 10000)
        b_g = proc.init_args.get("beta_decay_schedule", 10000)

        # Initialize subprocesses
        self.qC = QuadraticConnectivity(shape=shape_hess, hessian=hessian)

        if sparse:
            print("[INFO]: Using Sigma Delta Solution Neurons")
            self.sN = SigmaDeltaSolutionNeurons(
                shape=shape_sol,
                qp_neurons_init=qp_neuron_i,
                grad_bias=grad_bias,
                theta=theta,
                alpha=alpha,
                beta=beta,
                theta_decay_schedule=t_d,
                alpha_decay_schedule=a_d,
                beta_growth_schedule=b_g,
            )
            proc.vars.theta.alias(self.sN.vars.theta)
            proc.vars.theta_decay_schedule.alias(
                self.sN.vars.theta_decay_schedule
            )

        else:
            self.sN = SolutionNeurons(
                shape=shape_sol,
                qp_neurons_init=qp_neuron_i,
                grad_bias=grad_bias,
                alpha=alpha,
                beta=beta,
                alpha_decay_schedule=a_d,
                beta_growth_schedule=b_g,
            )
        self.cN = ConstraintNormals(
            shape=shape_constraint_matrix_T,
            constraint_normals=constraint_matrix_T,
        )

        # connect subprocesses to obtain required process behavior
        proc.in_ports.s_in.connect(self.cN.in_ports.s_in)
        self.cN.out_ports.a_out.connect(self.sN.in_ports.s_in_cn)
        self.sN.out_ports.a_out_qc.connect(self.qC.in_ports.s_in)
        self.qC.out_ports.a_out.connect(self.sN.in_ports.s_in_qc)
        self.sN.out_ports.a_out_cc.connect(proc.out_ports.a_out)

        # alias process variables to subprocess variables
        proc.vars.hessian.alias(self.qC.vars.weights)
        proc.vars.constraint_matrix_T.alias(self.cN.vars.weights)
        proc.vars.grad_bias.alias(self.sN.vars.grad_bias)
        proc.vars.qp_neuron_state.alias(self.sN.vars.qp_neuron_state)
        proc.vars.alpha.alias(self.sN.vars.alpha)
        proc.vars.beta.alias(self.sN.vars.beta)
        proc.vars.alpha_decay_schedule.alias(self.sN.vars.alpha_decay_schedule)
        proc.vars.beta_growth_schedule.alias(self.sN.vars.beta_growth_schedule)

        # profiling
        proc.vars.cN_synops.alias(self.cN.vars.synops)
        proc.vars.cN_neurops.alias(self.cN.vars.neurops)
        proc.vars.cN_spikeops.alias(self.cN.vars.spikeops)

        proc.vars.qC_synops.alias(self.qC.vars.synops)
        proc.vars.qC_neurops.alias(self.qC.vars.neurops)
        proc.vars.qC_spikeops.alias(self.qC.vars.spikeops)

        proc.vars.sN_synops.alias(self.sN.vars.synops)
        proc.vars.sN_neurops.alias(self.sN.vars.neurops)
        proc.vars.sN_spikeops.alias(self.sN.vars.spikeops)
