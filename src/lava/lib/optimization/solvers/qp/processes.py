# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort
import numpy as np
import typing as ty


class ConstraintDirections(AbstractProcess):
    """Connections in the constraint-checking group of neurons.
    Realizes the following abstract behavior:
    a_out = weights * s_in

    intialize the constraintDirectionsProcess

        Kwargs
        ------
        shape : int tuple, optional
            Define the shape of the connections matrix as a tuple. Defaults to
            (1,1)
        constraint_directions : (1-D  or 2-D np.array), optional
            Define the directions of the linear constraint hyperplanes. This is
            'A' in the constraints of the QP. Defaults to 0
    """

    def __init__(self, **kwargs: ty.Any):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1, 1))
        self.s_in = InPort(shape=(shape[1], 1))
        self.a_out = OutPort(shape=(shape[0], 1))
        weights = kwargs.pop("constraint_directions", 0)
        self.weights = Var(shape=shape, init=weights)

        # Profiling
        self.synops = Var(shape=(1, 1), init=0)
        self.neurops = Var(shape=(1, 1), init=0)
        self.spikeops = Var(shape=(1, 1), init=0)
        col_sum_init = np.count_nonzero(weights, axis=0)
        self.col_sum = Var(shape=col_sum_init.shape, init=col_sum_init)
        # self.col_sum = col_sum_init


class SigmaNeurons(AbstractProcess):
    """Process to accumate spikes into a state variable before being fed to
    another process.
    Realizes the following abstract behavior:
    a_out = self.x_internal + s_in

    Intialize the constraintNeurons Process.

        Kwargs:
        ------
        shape : int tuple, optional
            Define the shape of the thresholds vector. Defaults to (1,1).
        x_int_init : 1-D np.array, optional
            initial value of internal sigma neurons
    """

    def __init__(self, **kwargs: ty.Any):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1, 1))
        self.s_in = InPort(shape=(shape[0], 1))
        self.a_out = OutPort(shape=(shape[0], 1))
        self.x_internal = Var(shape=shape, init=kwargs.pop("x_int_init", 0))

        # Profiling
        self.synops = Var(shape=(1, 1), init=0)
        self.neurops = Var(shape=(1, 1), init=0)
        self.spikeops = Var(shape=(1, 1), init=0)


class ConstraintNeurons(AbstractProcess):
    """Process to check the violation of the linear constraints of the QP. A
    graded spike corresponding to the violated constraint is sent from the out
    port.

    Realizes the following abstract behavior:
    a_out = (s_in - thresholds) * (s_in < thresholds)

    Intialize the constraintNeurons Process.

        Kwargs:
        ------
        shape : int tuple, optional
            Define the shape of the thresholds vector. Defaults to (1,1).
        thresholds : 1-D np.array, optional
            Define the thresholds of the neurons in the
            constraint checking layer. This is usually 'k' in the constraints
            of the QP. Default value of thresholds is 0.
    """

    def __init__(self, **kwargs: ty.Any):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1, 1))
        self.s_in = InPort(shape=(shape[0], 1))
        self.a_out = OutPort(shape=(shape[0], 1))
        self.thresholds = Var(shape=shape, init=kwargs.pop("thresholds", 0))

        # Profiling
        self.synops = Var(shape=(1, 1), init=0)
        self.neurops = Var(shape=(1, 1), init=0)
        self.spikeops = Var(shape=(1, 1), init=0)


class QuadraticConnectivity(AbstractProcess):
    """The connections that define the Hessian of the quadratic cost function
    Realizes the following abstract behavior:
    a_out = weights * s_in

    Intialize the quadraticConnectivity process.

        Kwargs:
        ------
        shape : int tuple, optional
            A tuple defining the shape of the connections matrix. Defaults to
            (1,1).
        hessian : 1-D  or 2-D np.array, optional
            Define the hessian matrix ('Q' in the cost function of the QP) in
            the QP. Defaults to 0.
    """

    def __init__(self, **kwargs: ty.Any):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1, 1))
        self.s_in = InPort(shape=(shape[1], 1))
        self.a_out = OutPort(shape=(shape[0], 1))
        weights = kwargs.pop("hessian", 0)
        self.weights = Var(shape=shape, init=weights)

        # Profiling
        self.synops = Var(shape=(1, 1), init=0)
        self.neurops = Var(shape=(1, 1), init=0)
        self.spikeops = Var(shape=(1, 1), init=0)
        col_sum_init = np.count_nonzero(weights, axis=0)
        self.col_sum = Var(shape=col_sum_init.shape, init=col_sum_init)


class SolutionNeurons(AbstractProcess):
    """The neurons that evolve according to the constraint-corrected gradient
    dynamics.
    Implements the abstract behaviour
    qp_neuron_state += (-alpha * (s_in_qc + grad_bias) - beta * s_in_cn)

    Intialize the solutionNeurons process.

        Kwargs:
        -------
        shape : int tuple, optional
            A tuple defining the shape of the qp neurons. Defaults to (1,1).
        qp_neurons_init : 1-D np.array, optional
            initial value of qp solution neurons
        grad_bias : 1-D np.array, optional
            The bias of the gradient of the QP. This is the value 'p' in the
            QP definition.
        alpha : 1-D np.array, optional
            Defines the learning rate for gradient descent. Defaults to 1.
        beta : 1-D np.array, optional
            Defines the learning rate for constraint-checking. Defaults to 1.
        alpha_decay_schedule : int, optional
            The number of iterations after which one right shift operation
            takes place for alpha. Default intialization to a very high value
            of 10000.
        beta_growth_schedule : int, optional
            The number of iterations after which one left shift operation takes
            place for beta. Default intialization to a very high value of
            10000.
    """

    def __init__(self, **kwargs: ty.Any):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1, 1))
        # In/outPorts that come from/go to the quadratic connectivity process
        self.s_in_qc = InPort(shape=(shape[0], 1))
        self.a_out_qc = OutPort(shape=(shape[0], 1))
        # In/outPorts that come from/go to the constraint normals process
        self.s_in_cn = InPort(shape=(shape[0], 1))
        # OutPort for constraint checking
        self.a_out_cc = OutPort(shape=(shape[0], 1))
        self.qp_neuron_state = Var(
            shape=shape, init=kwargs.pop("qp_neurons_init", np.zeros(shape))
        )
        self.grad_bias = Var(
            shape=shape, init=kwargs.pop("grad_bias", np.zeros(shape))
        )
        self.alpha = Var(
            shape=shape, init=kwargs.pop("alpha", np.ones((shape[0], 1)))
        )
        self.beta = Var(
            shape=shape, init=kwargs.pop("beta", np.ones((shape[0], 1)))
        )
        self.alpha_decay_schedule = Var(
            shape=(1, 1), init=kwargs.pop("alpha_decay_schedule", 10000)
        )
        self.beta_growth_schedule = Var(
            shape=(1, 1), init=kwargs.pop("beta_growth_schedule", 10000)
        )
        self.decay_counter = Var(shape=(1, 1), init=0)
        self.growth_counter = Var(shape=(1, 1), init=0)

        # Momentum
        self.prev_qp_neuron_state = Var(shape=shape, init=np.zeros(shape))
        self.gamma_m = Var(shape=(1, 1), init=1)
        self.u_prev = Var(shape=(1, 1), init=0)

        # Profiling
        self.synops = Var(shape=(1, 1), init=0)
        self.neurops = Var(shape=(1, 1), init=0)
        self.spikeops = Var(shape=(1, 1), init=0)


class SigmaDeltaSolutionNeurons(AbstractProcess):
    """The neurons that evolve according to the constraint-corrected gradient
    dynamic along with sigma-delta coding
    Implements the abstract behaviour
    qp_neuron_state += (-alpha * (s_in_qc + grad_bias) - beta * s_in_cn)
    Send spike if (-alpha * (s_in_qc + grad_bias) - beta * s_in_cn) > threshold

    Intialize the solutionNeurons process.

        Kwargs:
        -------
        shape : int tuple, optional
            A tuple defining the shape of the qp neurons. Defaults to (1,1).
        qp_neurons_init : 1-D np.array, optional
            initial value of qp solution neurons
        grad_bias : 1-D np.array, optional
            The bias of the gradient of the QP. This is the value 'p' in the
            QP definition.
        theta : 1-D np.array, optional
            Defines the threshold for sigma-delta spiking. Defaults to 0.
        alpha : 1-D np.array, optional
            Defines the learning rate for gradient descent. Defaults to 1.
        beta : 1-D np.array, optional
            Defines the learning rate for constraint-checking. Defaults to 1.
        theta_decay_schedule : int, optional
            The number of iterations after which one right shift operation
            takes place for theta. Default intialization to a very high value
            of 10000.
        alpha_decay_schedule : int, optional
            The number of iterations after which one right shift operation
            takes place for alpha. Default intialization to a very high value
            of 10000.
        beta_growth_schedule : int, optional
            The number of iterations after which one left shift operation takes
            place for beta. Default intialization to a very high value of
            10000.
    """

    def __init__(self, **kwargs: ty.Any):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1, 1))
        # In/outPorts that come from/go to the quadratic connectivity process
        self.s_in_qc = InPort(shape=(shape[0], 1))
        self.a_out_qc = OutPort(shape=(shape[0], 1))
        # In/outPorts that come from/go to the constraint normals process
        self.s_in_cn = InPort(shape=(shape[0], 1))
        # OutPort for constraint checking
        self.a_out_cc = OutPort(shape=(shape[0], 1))
        self.qp_neuron_state = Var(
            shape=shape, init=kwargs.pop("qp_neurons_init", np.zeros(shape))
        )
        self.prev_qp_neuron_state = Var(shape=shape, init=np.zeros(shape))
        self.grad_bias = Var(
            shape=shape, init=kwargs.pop("grad_bias", np.zeros(shape))
        )
        self.theta = Var(
            shape=shape, init=kwargs.pop("theta", np.zeros((shape[0], 1)))
        )
        self.alpha = Var(
            shape=shape, init=kwargs.pop("alpha", np.ones((shape[0], 1)))
        )
        self.beta = Var(
            shape=shape, init=kwargs.pop("beta", np.ones((shape[0], 1)))
        )
        self.theta_decay_schedule = Var(
            shape=(1, 1), init=kwargs.pop("theta_decay_schedule", 10000)
        )
        self.alpha_decay_schedule = Var(
            shape=(1, 1), init=kwargs.pop("alpha_decay_schedule", 10000)
        )
        self.beta_growth_schedule = Var(
            shape=(1, 1), init=kwargs.pop("beta_growth_schedule", 10000)
        )
        self.decay_counter_theta = Var(shape=(1, 1), init=0)
        self.decay_counter = Var(shape=(1, 1), init=0)
        self.growth_counter = Var(shape=(1, 1), init=0)

        # Momentum
        self.gamma_m = Var(shape=(1, 1), init=1)
        self.u_prev = Var(shape=(1, 1), init=0)

        # Profiling
        self.synops = Var(shape=(1, 1), init=0)
        self.neurops = Var(shape=(1, 1), init=0)
        self.spikeops = Var(shape=(1, 1), init=0)


class QPTerLIFSolutionNeurons(AbstractProcess):
    """Implements Neurons that evolve using LIF dynamics but use ternary
    spikes i.e +1, -1 or 0 (no spike).

    Implements the abstract behaviour
        u = u * (1 - du)
        u += s_in_qc + s_in_cn
        v = v * (1 - dv) + u + bias
        s_out = (-1) * (v <= vth_lo) + (v >= vth_hi)
        v[s_out != 0] = 0

    Intialize the solutionNeurons process.

        Kwargs:
        -------
        shape : int tuple, optional
            A tuple defining the shape of the qp neurons. Defaults to (1,1).
        qp_neurons_init : 1-D np.array, optional
            initial value of qp solution neurons. This is the voltage v in the
            LIF implementation
        u_init: 1-D np.array, optional
            initial value of input current, u. Defaults to 0.
        vth_lo : 1-D np.array, optional
            Defines the lower threshold for spiking. Defaults to 10.
        vth_hi : 1-D np.array, optional
            Defines the upper threshold for spiking. Defaults to -10.
        grad_bias : 1-D np.array, optional
            The bias of the gradient of the QP. This is the value 'p' in the
            QP definition. For the neuron model this is the bias
        alpha : 1-D np.array, optional
            Defines the du parameter for current (inp). Default 0.
        beta : 1-D np.array, optional
            Defines the dv parameter for voltage (qp_neuron_state). Default 0.
        alpha_decay_schedule : int, optional
            The number of iterations after which one right shift operation
            takes place for alpha. Default intialization to a very high value
            of 100000. This option is not used for most cases
        beta_growth_schedule : int, optional
            The number of iterations after which one left shift operation takes
            place for beta. Default intialization to a very high value of
            100000. This option is not used for most cases
    """

    def __init__(self, **kwargs: ty.Any):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1, 1))
        # In/outPorts that come from/go to the quadratic connectivity process
        self.s_in_qc = InPort(shape=(shape[0], 1))
        self.a_out_qc = OutPort(shape=(shape[0], 1))
        # In/outPorts that come from/go to the constraint normals process
        self.s_in_cn = InPort(shape=(shape[0], 1))
        # OutPort for constraint checking
        self.a_out_cc = OutPort(shape=(shape[0], 1))
        self.qp_neuron_state = Var(
            shape=shape, init=kwargs.pop("qp_neurons_init", np.zeros(shape))
        )
        self.inp = Var(shape=shape, init=kwargs.pop("u_init", np.zeros(shape)))
        vth_hi = kwargs.pop("vth_hi", 10)
        vth_lo = kwargs.pop("vth_lo", -10)

        if vth_lo > vth_hi:
            raise AssertionError(
                f"Lower threshold {vth_lo} is larger than the "
                f"upper threshold {vth_hi} for Ternary LIF "
                f"neurons. Consider switching the values."
            )
        self.vth_lo = Var(shape=shape, init=vth_lo)
        self.vth_hi = Var(shape=shape, init=vth_hi)
        self.grad_bias = Var(
            shape=shape, init=kwargs.pop("grad_bias", np.zeros(shape))
        )
        self.alpha = Var(
            shape=shape, init=kwargs.pop("alpha", np.zeros((shape[0], 1)))
        )
        self.beta = Var(
            shape=shape, init=kwargs.pop("beta", np.zeros((shape[0], 1)))
        )
        self.alpha_decay_schedule = Var(
            shape=(1, 1), init=kwargs.pop("alpha_decay_schedule", 100000)
        )
        self.beta_growth_schedule = Var(
            shape=(1, 1), init=kwargs.pop("beta_growth_schedule", 100000)
        )
        self.decay_counter = Var(shape=(1, 1), init=0)
        self.growth_counter = Var(shape=(1, 1), init=0)

        # Profiling
        self.synops = Var(shape=(1, 1), init=0)
        self.neurops = Var(shape=(1, 1), init=0)
        self.spikeops = Var(shape=(1, 1), init=0)


class ConstraintNormals(AbstractProcess):
    """Connections influencing the gradient dynamics when constraints are
    violated.
    Realizes the following abstract behavior:
    a_out = weights * s_in

    Intialize the constraint normals to assign weights to constraint
    violation spikes.

        Kwargs:
        ------
        shape : int tuple, optional
            A tuple defining the shape of the connections matrix. Defaults to
            (1,1).
        constraint_normals : 1-D  or 2-D np.array
            Define the normals of the linear constraint hyperplanes. This is
            A^T in the constraints of the QP. Defaults to 0
    """

    def __init__(self, **kwargs: ty.Any):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1, 1))
        self.s_in = InPort(shape=(shape[1], 1))
        self.a_out = OutPort(shape=(shape[0], 1))
        weights = kwargs.pop("constraint_normals", 0)
        self.weights = Var(shape=shape, init=weights)

        # Profiling
        self.synops = Var(shape=(1, 1), init=0)
        self.neurops = Var(shape=(1, 1), init=0)
        self.spikeops = Var(shape=(1, 1), init=0)
        col_sum_init = np.count_nonzero(weights, axis=0)
        self.col_sum = Var(shape=col_sum_init.shape, init=col_sum_init)


class ConstraintCheck(AbstractProcess):
    """Check if linear constraints (equality/inequality) are violated for the
    qp. Recieves and sends graded spike from and to the gradientDynamics
    process. House the constraintDirections and constraintNeurons as
    sub-processes.

    Implements Abstract behavior:
    (constraint_matrix*x-constraint_bias)*(constraint_matrix*x<constraint_bias)

    Initialize constraintCheck Process.

        Kwargs:
        ------
        constraint_matrix : 1-D  or 2-D np.array, optional
            The value of the constraint matrix. This is 'A' in the linear
            constraints.
        constraint_bias : 1-D np.array, optional
            The value of the constraint bias. This is 'k' in the linear
            constraints.
        sparse: bool, optional
            Sparse is true when using sparsifying neuron-model eg. sigma-delta
        x_int_init : 1-D np.array, optional
            initial value of internal sigma neurons
    """

    def __init__(self, **kwargs: ty.Any):
        super().__init__(**kwargs)
        constraint_matrix = kwargs.pop("constraint_matrix", 0)
        shape = constraint_matrix.shape
        self.s_in = InPort(shape=(shape[1], 1))
        self.constraint_matrix = Var(shape=shape, init=constraint_matrix)
        self.constraint_bias = Var(
            shape=(shape[0], 1), init=kwargs.pop("constraint_bias", 0)
        )
        self.x_internal = Var(
            shape=(shape[1], 1), init=kwargs.pop("x_int_init", 0)
        )
        self.a_out = OutPort(shape=(shape[0], 1))

        # Profiling
        self.cNeur_synops = Var(shape=(1, 1), init=0)
        self.cNeur_neurops = Var(shape=(1, 1), init=0)
        self.cNeur_spikeops = Var(shape=(1, 1), init=0)

        self.cD_synops = Var(shape=(1, 1), init=0)
        self.cD_neurops = Var(shape=(1, 1), init=0)
        self.cD_spikeops = Var(shape=(1, 1), init=0)


class GradientDynamics(AbstractProcess):
    """Perform gradient descent with constraint correction to converge at the
    solution of the QP.

    Implements Abstract behavior:
    -alpha*(Q@x_init + p)- beta*A_T@graded_constraint_spike

    Initialize gradientDynamics Process.

        Kwargs:
        ------
        hessian : 1-D  or 2-D np.array, optional
            Define the hessian matrix ('Q' in the cost function of the QP) in
            the QP. Defaults to 0.
        constraint_matrix_T : 1-D  or 2-D np.array, optional
            The value of the transpose of the constraint matrix. This is 'A^T'
            in the linear constraints.
        grad_bias : 1-D np.array, optional
            The bias of the gradient of the QP. This is the value 'p' in the QP
            definition.
        qp_neurons_init : 1-D np.array, optional
            Initial value of qp solution neurons
        sparse: bool, optional
            Sparse is true when using sparsifying neuron-model eg. sigma-delta
        model: str, optional
            "SigDel" for sigma delta neurons and "TLIF" for Ternary LIF neurons.
            Defines the type of neuron to be used for sparse activity.
        vth_lo : 1-D np.array, optional
            Defines the lower threshold for TLIF spiking. Defaults to 10.
        vth_hi : 1-D np.array, optional
            Defines the upper threshold for TLIF spiking. Defaults to -10.
        theta : 1-D np.array, optional
            Defines the threshold for sigma-delta spiking. Defaults to 0.
        alpha : 1-D np.array, optional
            Define the learning rate for gradient descent. Defaults to 1.
        beta : 1-D np.array, optional
            Define the learning rate for constraint-checking. Defaults to 1.
        theta_decay_schedule : int, optional
            The number of iterations after which one right shift operation
            takes place for theta. Default intialization to a very high value
            of 10000.
        alpha_decay_schedule : int, optional
            The number of iterations after which one right shift operation
            takes place for alpha. Default intialization to a very high value
            of 10000.
        beta_growth_schedule : int, optional
            The number of iterations after which one left shift operation takes
            place for beta. Default intialization to a very high value of
            10000.
    """

    def __init__(self, **kwargs: ty.Any):
        super().__init__(**kwargs)
        hessian = kwargs.pop("hessian", 0)
        constraint_matrix_T = kwargs.pop("constraint_matrix_T", 0)
        shape_hess = hessian.shape
        shape_constraint_matrix_T = constraint_matrix_T.shape
        self.s_in = InPort(shape=(shape_constraint_matrix_T[1], 1))
        self.hessian = Var(shape=shape_hess, init=hessian)
        self.constraint_matrix_T = Var(
            shape=shape_constraint_matrix_T, init=constraint_matrix_T
        )
        self.grad_bias = Var(
            shape=(shape_hess[0], 1),
            init=kwargs.pop("grad_bias", np.zeros((shape_hess[0], 1))),
        )
        self.qp_neuron_state = Var(
            shape=(shape_hess[0], 1),
            init=kwargs.pop("qp_neurons_init", np.zeros((shape_hess[0], 1))),
        )

        self.vth_lo = Var(
            shape=(shape_hess[0], 1), init=kwargs.pop("vth_lo", -10)
        )
        self.vth_hi = Var(
            shape=(shape_hess[0], 1), init=kwargs.pop("vth_hi", 10)
        )

        self.theta = Var(
            shape=(shape_hess[0], 1),
            init=kwargs.pop("theta", np.zeros((shape_hess[0], 1))),
        )
        self.alpha = Var(
            shape=(shape_hess[0], 1),
            init=kwargs.pop("alpha", np.ones((shape_hess[0], 1))),
        )
        self.beta = Var(
            shape=(shape_hess[0], 1),
            init=kwargs.pop("beta", np.ones((shape_hess[0], 1))),
        )
        self.theta_decay_schedule = Var(
            shape=(1, 1), init=kwargs.pop("theta_decay_schedule", 10000)
        )
        self.alpha_decay_schedule = Var(
            shape=(1, 1), init=kwargs.pop("alpha_decay_schedule", 10000)
        )
        self.beta_growth_schedule = Var(
            shape=(1, 1), init=kwargs.pop("beta_growth_schedule", 10000)
        )

        self.a_out = OutPort(shape=(shape_hess[0], 1))

        # Profiling
        self.cN_synops = Var(shape=(1, 1), init=0)
        self.cN_neurops = Var(shape=(1, 1), init=0)
        self.cN_spikeops = Var(shape=(1, 1), init=0)

        self.qC_synops = Var(shape=(1, 1), init=0)
        self.qC_neurops = Var(shape=(1, 1), init=0)
        self.qC_spikeops = Var(shape=(1, 1), init=0)

        self.sN_synops = Var(shape=(1, 1), init=0)
        self.sN_neurops = Var(shape=(1, 1), init=0)
        self.sN_spikeops = Var(shape=(1, 1), init=0)


class OutProbeProcess(AbstractProcess):
    def __init__(self, **kwargs):
        """Use to set read output spike from a process

        Kwargs:
        ------
            out_shape : int tuple, optional
                Set OutShape to custom value
        """
        super().__init__(**kwargs)
        shape = kwargs.pop("out_shape", (1, 1))
        iterations = kwargs.pop("iterations", 1)
        self.s_in = InPort(shape=shape)
        var_shape = (iterations, shape[0])
        self.sol_list = Var(shape=var_shape, init=np.zeros(var_shape))
        self.it_counter = Var(shape=(1, 1), init=0)
