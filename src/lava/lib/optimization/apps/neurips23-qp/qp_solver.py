import numpy as np
from scipy import sparse
import osqp
from dataclasses import dataclass
import typing as ty
import numpy.typing as npty
from matplotlib import rcParams, cycler
import matplotlib.pyplot as plt


def convert_to_fp(mat, man_bits):
    """Function that returns the exponent, mantissa representation for
    floating point numbers that need to be represented on Loihi. A global exp
    is calculated for the matrices based on the max absolute value in the
    matrix. This is then used to calculate the manstissae in the matrix.

    Args:
        mat (np.float): The input floating point matrix that needs to be
        converted
        man_bits (int): number of bits that the mantissa uses for it's
        representation (Including sign)

    Returns:
        mat_fp_man (np.int): The matrix in
    """
    exp = np.ceil(np.log2(np.max(np.abs(mat)))) - man_bits + 1
    mat_fp_man = (mat // 2**exp).astype(int)
    return mat_fp_man.astype(int), exp.astype(int)


def ruiz_equilibriation(matrix, iterations):
    """Preconditioning routine used to make the first-order QP solver converge
    faster. Returns preconditioners to be used to operate on matrices of the QP.

    Args:
        matrix (np.float): Matrix that needs to be preconditioned
        iterations (int): Number of iterations that preconditioner runs for

    Returns:
        left_preconditioner (np.float): The left preconditioning matrix
        right_preconditioner (np.float): The right preconditioning matrix

    """
    m_bar = matrix
    left_preconditioner = sparse.csc_matrix(np.eye(matrix.shape[0]))
    right_preconditioner = sparse.csc_matrix(np.eye(matrix.shape[1]))
    for _ in range(iterations):
        D_l_inv = sparse.csc_matrix(
            np.diag(1 / np.sqrt(np.linalg.norm(m_bar, ord=2, axis=1)))
        )
        if m_bar.shape[0] != m_bar.shape[1]:
            D_r_inv = sparse.csc_matrix(
                np.diag(1 / np.sqrt(np.linalg.norm(m_bar, ord=2, axis=0)))
            )
        else:
            D_r_inv = D_l_inv

        m_bar = D_l_inv @ m_bar @ D_r_inv
        left_preconditioner = left_preconditioner @ D_l_inv
    return left_preconditioner, right_preconditioner


def get_complete_filenames_with_path():
    # problem_file = "NeuRIPS_data/anymal_quadruped/01_anymal_small_scale/mpc_240_0800.npz"
    path_prefix = "NeuRIPS_data/anymal_quadruped/"
    size_indices = list(range(1, 7))
    dir_prefixes = ['0' + str(idx) for idx in size_indices]
    size_qualitv = ['small', 'medium', 'medium_large', 'full', 'extra_full', '2x_full']
    prefix_szql = list(zip(dir_prefixes, size_qualitv))
    dir_names = [path_prefix + prefix + '_anymal_' + szql + '_scale/' for (prefix, szql) in prefix_szql]
    size_quantiv = [240, 2400, 3600, 4800, 7200, 8400]
    file_name_prefixes = ['mpc_' + str(szqn) + '_' for szqn in size_quantiv]
    dir_file_pfx = list(zip(dir_names, file_name_prefixes))
    file_path_pfx = [dn + fnp for (dn, fnp) in dir_file_pfx]
    problem_ids = ['0150', '0300', '0500', '0700', '0800', '0900', '1200', '1500', '1800', '2000']
    file_names = [fppfx + pid + '.npz' for fppfx in file_path_pfx for pid in problem_ids]
    
    return file_names


@dataclass
class PerturbationSettings:
    do_perturb: bool = False
    noise_ampl: float = 0.0


class QPProblemNumPy:

    def __init__(self, problem_file, perturb_settings = PerturbationSettings()):
        self.problem_file = problem_file
        self.perturb_settings = perturb_settings
        
        self.Q = None
        self.p = None
        self.A = None
        self.k = None

        self.Q_pre = None
        self.p_pre = None
        self.A_pre = None
        self.k_pre = None

        self.pre_mat_Q = None
        self.pre_mat_A = None
        self.Q_pre_fp_man = None
        self.Q_pre_fp_exp = None
        self.p_pre_fp_man = None
        self.p_pre_fp_exp = None
        self.A_pre_fp_man = None
        self.A_pre_fp_exp = None
        self.k_pre_fp_man = None
        self.k_pre_fp_exp = None

        self.Q_exp_new = None
        self.A_exp_new = None
        self.correction_exp = None

        self.load_matrices(problem_file)
        if perturb_settings.do_perturb:
            self.Q_orig = self.Q.copy()
            self.A_orig = self.A.copy()
            Q_per = self.Q.copy()
            A_per = self.A.copy()
            vari = perturb_settings.noise_ampl
            Q_per[np.nonzero(Q_per)] = \
                np.random.normal(self.Q[np.nonzero(self.Q)], 
                                 vari * np.abs(self.Q[np.nonzero(self.Q)]),
                                 len(np.nonzero(self.Q)[0]))
            A_per[np.nonzero(A_per)] = \
                np.random.normal(self.A[np.nonzero(self.A)],
                                 vari * np.abs(self.A[np.nonzero(self.A)]),
                                 len(np.nonzero(self.A)[0]))
            self.Q = Q_per.copy()
            self.A = A_per.copy()
        self.precond_matrices()
        self.fixed_point_adjustment()

    def load_matrices(self, filename):
        # Load problem matrices
        matrices = np.load(filename)
        self.Q, self.A, self.p, self.k = [matrices[i] for i in matrices]
        self.k = self.k.reshape(self.k.shape[0], )
        self.p = self.p.reshape(self.p.shape[0], )
        
    def precond_matrices(self):
        # Precondition
        (self.pre_mat_Q, _) = ruiz_equilibriation(self.Q, 5)
        self.Q_pre = self.pre_mat_Q @ self.Q @ self.pre_mat_Q
        self.p_pre = self.pre_mat_Q @ self.p

        (self.pre_mat_A, _) = ruiz_equilibriation(self.A, 5)
        self.A_pre = self.pre_mat_A @ self.A @ self.pre_mat_Q
        self.k_pre = self.pre_mat_A @ self.k

    def fixed_point_adjustment(self):
        # Fixed point conversion
        self.k_pre_fp_man, self.k_pre_fp_exp = self.k_pre.astype(int), 0
        self.Q_pre_fp_man, self.Q_pre_fp_exp = convert_to_fp(self.Q_pre, 8)
        self.A_pre_fp_man, self.A_pre_fp_exp = convert_to_fp(self.A_pre, 8)
        self.p_pre_fp_man, self.p_pre_fp_exp = convert_to_fp(self.p_pre, 16)

        self.correction_exp = -min(self.A_pre_fp_exp, self.Q_pre_fp_exp)
        self.Q_exp_new, self.A_exp_new = (
            self.correction_exp + self.Q_pre_fp_exp,
            self.correction_exp + self.A_pre_fp_exp,
        )
        self.A_pre_fp_man = (self.A_pre_fp_man // 2) * 2
        self.Q_pre_fp_man = (self.Q_pre_fp_man // 2) * 2


class QPSolverNumPy:

    def __init__(self,
                 qpp: QPProblemNumPy,
                 num_iter: int = 300,
                 init_state_x=None,
                 init_state_w=None):
        self.problem = qpp
        self.alpha_man = None
        self.alpha_exp = None
        self.beta_man = None
        self.beta_exp = None
        self. alpha_decay_indices_list = (
                np.array([50, 100, 200, 350, 550, 800, 1100, 1450, 1850,
                          2300]) * 2
        )
        self.beta_growth_indices_list = (
                np.array([1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2045,
                          4095]) * 2 + 1
        )

        self.state_var_w_py = None
        self.state_var_x_py = None

        self.iterations = num_iter
        self.growth_counter = 0
        self.decay_counter = 0
        self.gamma_py = np.zeros(qpp.k.shape).astype(np.int32)

        self.cost = []
        self.constraint_sat = []
        if init_state_x is None:
            self.init_state_x = np.zeros(qpp.p.shape).astype(np.int32)
        else:
            self.init_state_x = init_state_x
        if init_state_x is None:
            self.init_state_w = np.zeros(qpp.k.shape).astype(np.int32)
        else:
            self.init_state_w = init_state_w

        self.osqp_res = None

    def initialize_state(self):
        # Assign init cond to state var
        self.state_var_x_py = self.init_state_x
        self.state_var_w_py = self.init_state_w + np.right_shift(
            self.beta_man * (-np.right_shift(self.problem.k_pre_fp_man,
                                             -self.problem.k_pre_fp_exp)),
            -self.beta_exp
        )

    def reset_alpha_beta(self):
        self.alpha_man = 160
        self.alpha_exp = -8
        self.beta_man = 7
        self.beta_exp = -10

    def solve(self):
        self.reset_alpha_beta()
        self.initialize_state()
        # Solution loop
        print("Solving.. ", end="", flush=True)
        for i in range(1, self.iterations):
            a_in_pg_1 = self.problem.A_pre_fp_man.T @ self.gamma_py
            a_in_pg_1 = np.left_shift(a_in_pg_1, self.problem.A_exp_new)
            a_in_pg_2 = self.problem.Q_pre_fp_man @ self.state_var_x_py
            a_in_pg_2 = np.left_shift(a_in_pg_2, self.problem.Q_exp_new)

            a_in_pg = a_in_pg_1 + a_in_pg_2
            a_in_pg = np.right_shift(a_in_pg, self.problem.correction_exp)

            self.decay_counter += 1
            if self.decay_counter % 2 == 0:
                if self.decay_counter in self.alpha_decay_indices_list:
                    self.alpha_man //= 2
                tot_bias_gd = self.alpha_man * (
                    a_in_pg + np.right_shift(self.problem.p_pre_fp_man,
                                             -self.problem.p_pre_fp_exp)
                )

                x_inter = np.right_shift(tot_bias_gd, -self.alpha_exp)

                self.state_var_x_py -= x_inter

                # Logging data to plot later
                curr_post_sol = self.problem.pre_mat_Q @ self.state_var_x_py
                self.cost.append(
                    curr_post_sol.T @ self.problem.Q @ curr_post_sol/2 +
                    self.problem.p.T @ curr_post_sol)
                self.constraint_sat.append(np.linalg.norm(
                    self.problem.A @ curr_post_sol - self.problem.k))
                # sol_list.append(state_var_x_py)
            a_in_pi = self.problem.A_pre_fp_man @ self.state_var_x_py
            a_in_pi = np.right_shift(a_in_pi, -self.problem.A_pre_fp_exp)

            self.growth_counter += 1
            if self.growth_counter % 2 == 1:
                if self.growth_counter in self.beta_growth_indices_list:
                    self.beta_man *= 2
                tot_bias_pi = self.beta_man * (
                    a_in_pi - np.right_shift(self.problem.k_pre_fp_man,
                                             -self.problem.k_pre_fp_exp)
                )

                omega = np.right_shift(tot_bias_pi, -self.beta_exp)
                self.state_var_w_py += omega
                self.gamma_py = self.state_var_w_py + omega
        print("Done", flush=True)

    def osqp_solve(self, 
                   verbose=False, 
                   initstatex=None, 
                   initstatew=None):
        # OSQP
        prob = osqp.OSQP()
        prob.setup(sparse.csc_matrix(self.problem.Q),
                   self.problem.p,
                   sparse.csc_matrix(self.problem.A),
                   self.problem.k,
                   self.problem.k,
                   verbose=verbose)
        if initstatex is not None and initstatew is not None:
            prob.warm_start(x=initstatex, y=initstatew)
        self.osqp_res = prob.solve()
