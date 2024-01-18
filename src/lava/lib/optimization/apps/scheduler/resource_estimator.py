# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import numpy
import numpy as np
from lava.lib.optimization.apps.scheduler.problems import (
    SatelliteScheduleProblem)


class ResourceEstimatorSatScheduler:

    def __init__(self,
                 ssp: SatelliteScheduleProblem,
                 num_wgt_bits: int,
                 neurons_per_core: int,
                 num_outports_per_neuron: int = 2,
                 max_neurons_per_core: int = 4096,
                 max_mem_words_per_core_mpds: int = int(2**14),
                 max_axon_mem_to_syn_mem_ratio: float = 0.1,
                 axon_mem_reduction_factor: float = 0.5
                 ):
        """

        Parameters
        ----------
        ssp (SatelliteSchedulingProblem) : Satellite scheduling problem for
        which resource estimation is being done.
        num_wgt_bits (int) : synaptic precision in bits.
        neurons_per_core (int) : initial user-specified number of neurons
        placed on each neurocore.
        max_neurons_per_core (int) : maximum number of
        max_synmem_words_per_core
        axon_mem_reduction_factor
        """
        self._problem = ssp
        self._num_wgt_bits = num_wgt_bits
        self._num_neurons_per_core = neurons_per_core
        self._num_outports_per_neuron = num_outports_per_neuron
        self._max_neurons_per_core = max_neurons_per_core
        self._max_mem_words_per_core_mpds = max_mem_words_per_core_mpds
        self._axon_mem_reduction_factor = axon_mem_reduction_factor
        self._max_axon_mem_to_syn_mem_ratio = max_axon_mem_to_syn_mem_ratio

        if not self._problem.graph:
            self._problem.generate()

        self.num_neurons = self._problem.graph.number_of_nodes()
        if self.num_neurons < self._num_neurons_per_core:
            self._num_neurons_per_core = self.num_neurons

        # Parameters derived from the SS problem
        self._num_satellites: int = ssp.num_satellites
        self._num_requests = ssp.num_requests
        self._delta_y_max = ssp.view_height
        self._delta = ssp.sat_spacing
        self._turn_rate = ssp.turn_rate

        # populate self.num_nodes_per_sat:
        self._compute_num_nodes_per_sat()
        # populate self.m_arr:
        self._compute_m_array()
        # Get an array of sat IDs in nodes, such that sat_ids[node=k] = ID of
        # satellite in node k
        self.sat_ids = self.sat_ids_in_nodes()
        # populate self.lower_bound_of_overlap_sat_ids_array and
        # self.upper_bound_of_overlap_sat_ids_array:
        self._compute_overlap_bounds_sat_ids()
        # compute number of cores
        self.num_cores = np.ceil(self.num_neurons /
                                 self.num_neurons_per_core).astype(int)
        # populate self.num_synapses_in_a_row and self.num_synapses_per_core
        self._compute_num_synapses_in_a_row()
        self._compute_synaptic_usage()
        # populate self.num_input_axons_per_core and
        # self.num_output_axons_per_core
        self._compute_axonal_usage()

    @property
    def problem(self):
        return self._problem

    @property
    def num_wgt_bits(self):
        return self._num_wgt_bits

    @property
    def num_neurons_per_core(self):
        return self._num_neurons_per_core

    @num_neurons_per_core.setter
    def num_neurons_per_core(self, val: int):
        self._num_neurons_per_core = val

    @property
    def num_outports_per_neuron(self):
        return self._num_outports_per_neuron

    @property
    def max_neurons_per_core(self):
        return self._max_neurons_per_core

    @property
    def max_mem_words_per_core_mpds(self):
        return self._max_mem_words_per_core_mpds

    @property
    def axon_mem_reduction_factor(self):
        return self._axon_mem_reduction_factor

    @property
    def max_axon_mem_to_syn_mem_ratio(self):
        return self._max_axon_mem_to_syn_mem_ratio

    @property
    def num_satellites(self):
        return self._num_satellites

    @property
    def num_requests(self):
        return self._num_requests

    @property
    def delta_y_max(self):
        return self._delta_y_max

    @property
    def delta(self):
        return self._delta

    @property
    def turn_rate(self):
        return self._turn_rate

    @property
    def num_satellites_in_overlap_list(self):
        """Computes and returns a list, whose jth entry is the number of
        satellites with an overlapping viewing window with satellite ID = j.

        Assuming equally spaced satellites, all satellites from lower to
        upper IDs are contiguously within the overlap zone of a given
        satellite.
        """
        yuu = self.upper_bound_of_overlap_sat_ids_array
        ell = self.lower_bound_of_overlap_sat_ids_array
        return yuu - ell + 1

    @property
    def num_syn_mem_words_per_core(self):
        # ToDo: This is an approximate estimate, assuming 8 bytes per syn mem
        #  word.
        return (self.num_wgt_bits * self.num_synapses_per_core // 64).astype(
            int)

    @property
    def num_axon_mem_words_per_core(self):
        axon_mem_words_needed = (self.axon_mem_reduction_factor *
                                 self.num_output_axons_per_core).flatten()
        words_in_axon_map = np.ceil(self.num_axon_map_entries_per_core * 36 /
                                    64).astype(int).flatten()
        result = np.zeros((self.num_cores, 1))
        if np.any(axon_mem_words_needed > words_in_axon_map):
            result[axon_mem_words_needed > words_in_axon_map] = np.ceil(
                axon_mem_words_needed - words_in_axon_map).astype(int)
        return result

    @property
    def num_syn_map_enries_per_core(self):
        return self.num_input_axons_per_core

    @property
    def num_axon_map_entries_per_core(self):
        return (self.num_outports_per_neuron * self.num_neurons_per_core *
                np.ones((self.num_cores, 1), dtype=int))

    @property
    def num_dend_accums_per_neuron(self):
        """Compute how many DAs are needed per neuron.

        Assumes that the entire fan-in of a neuron sends fully saturated
        activation of 2 ** (num_wgt_bits - 1).

        Notes
        -----
        Expected value of this parameter is 1 for all neurons, to satisfy QUBO
        neuron model. If for any neuron the value is > 1, then there are
        chances of DA overflow at execution time.
        """
        max_activation_per_neuron = (
                2 ** (self.num_wgt_bits - 1) * self.num_synapses_in_a_row)
        num_bits_in_max_activation = np.ceil(np.log2(max_activation_per_neuron))
        return np.ceil(num_bits_in_max_activation / 32).astype(int)

    def _compute_num_nodes_per_sat(self):
        """Computes how many graph nodes contain a particular satellite.
        """
        node_dict = dict(self._problem.graph.nodes)
        node_dict_sorted_by_sat_id = (
            dict(sorted(node_dict.items(), key=lambda item: item[1][
                "agent_id"])))
        sat_ids_multiplicity = np.array([v["agent_id"] for v in
                                         node_dict_sorted_by_sat_id.values()])
        sat_ids, self.num_nodes_per_sat = np.unique(sat_ids_multiplicity,
                                                    return_counts=True)
        # Sanity check:
        if self.num_satellites != self.num_nodes_per_sat.size:
            raise AssertionError("Number of satellites did not match the "
                                 "size of the array that counts the "
                                 "multiplicity of each satellite in the "
                                 "problem graph. Something must have gone "
                                 "wrong. Perhaps while generating the "
                                 "problem graph.")

    def _compute_m_array(self):
        """Computes cumulative sum of number of nodes per satellite.

        Given the Q-matrix corresponding to the Sat Scheduling problem,
        the values in the m-array are the row (or column) IDs where block
        matrices corresponding to individual satellites begin.
        """
        self.m_arr = np.zeros((self.num_satellites + 1, 1), dtype=int)
        cs = np.cumsum(self.num_nodes_per_sat)
        self.m_arr[1:] = cs.reshape((self.num_satellites, 1)).astype(int)

    def _compute_overlap_bounds_sat_ids(self):
        """Compute lower and upper satellite IDs for each satellite,
        which have overlapping viewing window with it.

        Assuming equally spaced satellites, all satellites from lower to
        upper IDs are contiguously within the overlap zone of a given
        satellite.
        """
        if np.isclose(self.delta_y_max, self.delta):
            half_num_ids = 0
        else:
            half_num_ids = np.ceil(self.delta_y_max / (2 * self.delta))
        self.lower_bound_of_overlap_sat_ids_array = (
            np.array([max(1, sat_id - half_num_ids) for sat_id in
                      self._problem.satellites])).astype(int)
        self.upper_bound_of_overlap_sat_ids_array = (
            np.array([min(sat_id + half_num_ids, self.num_satellites) for
                      sat_id in self._problem.satellites])).astype(int)

    def _compute_num_synapses_in_a_row(self):

        # Per row non-zero number of weights from the adjacency matrix + 1
        # for the diagonal element.
        self.num_synapses_in_a_row = np.count_nonzero(self.problem.adjacency,
                                                      axis=1, keepdims=True) + 1

        # yuu_arr = self.upper_bound_of_overlap_sat_ids_array
        # ell_arr = self.lower_bound_of_overlap_sat_ids_array
        # self.num_synapses_in_a_row = np.zeros((self.num_neurons, 1),
        #                                       dtype=int)
        # # ToDo: Vectorize this for entire vector of num_neurons dimension
        # for p in range(self.num_neurons):
        #     j = self.sat_ids[p]
        #     start_idx = self.m_arr[ell_arr[j] - 1]
        #     end_idx = self.m_arr[yuu_arr[j]]
        #     self.num_synapses_in_a_row[p] = end_idx - start_idx

    def _compute_synaptic_usage(self):
        """Compute the number of synapses required for each neuron, i.e.,
        per row (equivalently, per column) of the Q-matrix and using that,
        compute the number of synapses required on each core.
        """

        self.num_synapses_per_core = np.zeros((self.num_cores, 1),
                                              dtype=int)
        # ToDo: Vectorize this for entire vector of num_cores dimension
        for core_id in range(self.num_cores):
            start_idx = int(core_id * self.num_neurons_per_core)
            end_idx = int((core_id + 1) * self.num_neurons_per_core)
            self.num_synapses_per_core[core_id] = np.sum(
                self.num_synapses_in_a_row[start_idx:end_idx])

    def _compute_axonal_usage(self):
        self.num_input_axons_per_core = np.zeros((self.num_cores, 1), dtype=int)
        self.num_output_axons_per_core = np.zeros((self.num_cores, 1),
                                                  dtype=int)
        yuu_arr = self.upper_bound_of_overlap_sat_ids_array
        ell_arr = self.lower_bound_of_overlap_sat_ids_array
        # ToDo: Vectorize this for entire vector of num_cores dimension
        for core_id in range(self.num_cores):
            start_idx = core_id * self.num_neurons_per_core
            end_idx = (core_id + 1) * self.num_neurons_per_core
            ubounds = np.zeros((self.num_neurons_per_core, 1), dtype=int)
            lbounds = np.zeros((self.num_neurons_per_core, 1), dtype=int)
            for rowid in range(self.num_neurons_per_core):
                sat_id = self.sat_ids[rowid]
                ubounds[rowid] = self.m_arr[yuu_arr[sat_id]]
                lbounds[rowid] = self.m_arr[ell_arr[sat_id] - 1]
            sup_ubound = np.max(ubounds)
            inf_lbound = np.min(lbounds)
            self.num_input_axons_per_core[core_id] = int(sup_ubound -
                                                         inf_lbound)
            # np.max(self.num_synapses_in_a_row[start_idx:end_idx]).astype(int)
            self.num_output_axons_per_core[core_id] = np.ceil(
                int(sup_ubound - inf_lbound) / self.num_cores).astype(int)
            # self.num_synapses_in_a_row[start_idx:end_idx]

    def sat_ids_in_nodes(self) -> numpy.ndarray:
        """Computes the satellite IDs in each node.

        Returns
        -------
        sat_ids (numpy.ndarray): a vector of IDs of the satellites for each
        node.
        """

        p_arr = np.arange(self.num_neurons)
        p_arr = np.tile(p_arr, (self.m_arr.size - 1, 1))

        ranges = np.hstack((self.m_arr[:-1], self.m_arr[1:]))
        lower = ranges[:, 0].reshape(len(ranges), 1) <= p_arr
        upper = p_arr < ranges[:, 1].reshape(len(ranges), 1)

        indices = np.nonzero(np.logical_and(lower, upper))
        return indices[0]

    def core_ids_for_sat(self, sat_id: int) -> np.ndarray:
        """Computes the core ID(s) over which the nodes (i.e., neurons)
        corresponding to a satellite with `sat_id` are located.

        Notes
        -----
        Reminder: `sat_id` begins at 1 and ends at N_{sat}.
        """
        lower = np.floor(self.m_arr[sat_id-1] / self.num_neurons_per_core)
        upper = np.floor(self.m_arr[sat_id] - 1 / self.num_neurons_per_core) + 1

        return np.arange(lower, upper)

    def _compute_targets(self):
        target_num_axon_mem_words = np.ceil(
            self.max_axon_mem_to_syn_mem_ratio *
            self.num_syn_mem_words_per_core).astype(int)
        target_num_output_axons = (
            ((1 / self.axon_mem_reduction_factor) *
             target_num_axon_mem_words)).astype(int)
        target_syn_map_entries = target_num_output_axons
        target_axon_map_entries = self.num_neurons_per_core

        target_num_words = (self.num_syn_mem_words_per_core +
                            target_num_axon_mem_words + (
                                    target_syn_map_entries / 2) +
                            target_axon_map_entries).astype(int)
        return (target_num_axon_mem_words, target_syn_map_entries,
                target_axon_map_entries, target_num_words)

    def partition(self):
        iter_counter = 0
        while True:
            iter_counter += 1
            if iter_counter >= 1000:
                print(f"Iteration: {iter_counter}")
                break
            # if self.num_neurons_per_core > self.num_neurons:
            #     self.num_neurons_per_core = self.num_neurons
            #     self.num_cores = np.ceil(self.num_neurons /
            #                              self.num_neurons_per_core).astype(int)
            #     self._compute_synaptic_usage()
            #     self._compute_axonal_usage()
            #     break

            if self.num_neurons_per_core > self.max_neurons_per_core:
                self.num_neurons_per_core -= 1
                self.num_cores = np.ceil(self.num_neurons /
                                         self.num_neurons_per_core).astype(int)
                self._compute_synaptic_usage()
                self._compute_axonal_usage()

            (target_num_axon_mem_words, target_syn_map_entries,
             target_axon_map_entries, target_num_words) = (
                self._compute_targets())

            if np.any(target_num_words > self.max_mem_words_per_core_mpds):
                self.num_neurons_per_core -= 1
                if self.num_neurons_per_core <= 0:
                    raise AssertionError("num_neurons_per_core cannot be zero. "
                                         "Network is likely too big and "
                                         "exceeds per-core memory capacity "
                                         "for synapses.")
                self.num_cores = np.ceil(self.num_neurons /
                                         self.num_neurons_per_core).astype(int)
                self._compute_synaptic_usage()
                self._compute_axonal_usage()
                (target_num_axon_mem_words, target_syn_map_entries,
                 target_axon_map_entries, target_num_words) = (
                    self._compute_targets())

            if (np.any(self.num_axon_mem_words_per_core >
                       target_num_axon_mem_words)):
                self.num_neurons_per_core += 1
                self.num_cores = np.ceil(self.num_neurons /
                                         self.num_neurons_per_core).astype(int)
                self._compute_synaptic_usage()
                self._compute_axonal_usage()
                (target_num_axon_mem_words, target_syn_map_entries,
                 target_axon_map_entries, target_num_words) = (
                    self._compute_targets())

            total_num_words = (self.num_syn_mem_words_per_core +
                               self.num_axon_mem_words_per_core + (
                                   self.num_syn_map_enries_per_core / 2) +
                               self.num_axon_map_entries_per_core)
            if (np.all(total_num_words <= self.max_mem_words_per_core_mpds) and
                    np.all(self.num_axon_mem_words_per_core <=
                           target_num_axon_mem_words) and
                    self.num_neurons_per_core <= self.max_neurons_per_core):
                break
