from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import networkx as nx
from collections import deque
import copy

import tempfile

import numpy as np
from tf_agents.environments import py_environment

from tf_agents.specs import array_spec

from tf_agents.trajectories import time_step as ts

from initialize import SFC

tempdir = tempfile.gettempdir()

init = SFC()
phy_nodes = init.physical_node()
sfc_map, sfc_th, availability, availability_matrix = init.random_sfc()
transition_prob = np.random.rand(sfc_map.shape[0], sfc_map.shape[1], 2)


############################# SFC RUN-TIME ####################################
class SFC_run(py_environment.PyEnvironment):

    def __init__(self):

        super().__init__()

        self.VNF_state = np.zeros((sfc_map.shape))
        self.trans_step = np.zeros((sfc_map.shape))
        self.posible_states = np.array([0, 1, 2])
        self.phy_nodes = copy.deepcopy(phy_nodes)
        self.phy_nodes_init = copy.deepcopy(phy_nodes)
        self.sfc_map = sfc_map
        self.th = sfc_th
        self.transition_prob = transition_prob
        self._state = None
        self.th_store = np.zeros((sfc_th.shape))
        self.failure = np.zeros((self.sfc_map.shape))
        self.fail_th = np.array([2, 4, 5])
        self.action_index = None
        self.backup_node_req = [50, 70, 100]
        self.backup_bw_req = [50, 100]
        self.backup_map = np.full((self.sfc_map.shape), -1)
        self.backup_map_init = copy.deepcopy(self.backup_map)
        self.backup_availability = np.zeros((self.sfc_map.shape))
        self.backup_availability_init = copy.deepcopy(self.backup_availability)
        self.paths = None
        self.failure_protocol_map = np.zeros(self.sfc_map.shape)
        self.last_chosen_path = []
        self.borrowed_bw = []
        self.step_num = 0
        self._episode_ended = False
        self.episode_length = 200
        self.availability_matrix = copy.deepcopy(availability_matrix)
        self.availability_req = copy.deepcopy(availability)
        self.monitor_mode = False
        self.lost = 0
        self.wrong_backup_action = 0
        self.wrong_fail_action = 0
        self.not_enough_backup_resource = False

        self._action_spec = array_spec.BoundedArraySpec \
            (shape=(self.GetActionDimension()[0] * self.GetActionDimension()[1],), dtype=np.float32,
             minimum=-1.0,
             maximum=1.0, name='action')
        self._observation_spec = array_spec.BoundedArraySpec \
            (shape=(self.GetObservationDimension(),), dtype=np.float32, minimum=0.0, maximum=2, name='observation')

    ############################################################################################################

    def markov_trans(self):
        self.lost = 0
        for idx, state in np.ndenumerate(self.VNF_state):
            if state == 0:
                self.VNF_state[idx] = np.random.choice([0, 1],
                                                       p=[1 - (self.transition_prob[idx[0], idx[1], 0]),
                                                          self.transition_prob[idx[0], idx[1], 0]])
                self.trans_step[idx] = 1 if self.VNF_state[idx] == 1 else 0

            elif state == 1:
                if self.trans_step[idx] > 2:
                    p2 = self.transition_prob[idx[0], idx[1], 1] * (self.trans_step[idx] - 2)
                    if p2 >= 1:
                        p2 = 1
                    self.VNF_state[idx] = np.random.choice([0, 1, 2], p=[(1 - p2) / 2, (1 - p2) / 2, p2])
                else:
                    self.trans_step[idx] += 1
            elif state == 2:

                self.VNF_state[idx] = 0
                self.lost -= 10

    def backup_placement(self, action_idx):
        chosen_node = None

        if self.backup_map[action_idx] == -1:
            self.wrong_backup_action = 0

            cpu_cap_dict = nx.get_node_attributes(self.phy_nodes, "CPU")
            self.sorted_CPU = sorted(cpu_cap_dict, key=cpu_cap_dict.get, reverse=True)
            i = 0
            done = False
            self.not_enough_backup_resource = False

            while not done:  #### choosing backup nodewith greedy policy on CPU capacity
                if i < len(self.sorted_CPU) and self.sorted_CPU[i] == self.sfc_map[action_idx]:
                    i += 1
                else:
                    #### backup node must not be the mapping node

                    if i < len(self.sorted_CPU):
                        if self.phy_nodes.nodes[self.sorted_CPU[i]]['CPU'] - self.backup_node_req[0] >= 0 and \
                                self.phy_nodes.nodes[self.sorted_CPU[i]]['storage'] - self.backup_node_req[1] >= 0 and \
                                self.phy_nodes.nodes[self.sorted_CPU[i]]['memory'] - self.backup_node_req[2] >= 0 and \
                                self.phy_nodes.edges[self.sfc_map[action_idx], self.sorted_CPU[i]]['bw'] - \
                                self.backup_bw_req[0] >= 0:
                            done = True
                            self.not_enough_backup_resource = False
                            chosen_node = self.sorted_CPU[i]

                        else:
                            done = False



                    else:
                        self.not_enough_backup_resource = True

                        done = True

                i += 1

                # negative reward
            if self.not_enough_backup_resource:
                pass


            else:
                reward = +10
                self.backup_map[action_idx] = chosen_node

                self.phy_nodes.nodes[chosen_node]['CPU'] -= self.backup_node_req[0]
                self.phy_nodes.nodes[chosen_node]['storage'] -= self.backup_node_req[1]
                self.phy_nodes.nodes[chosen_node]['memory'] -= self.backup_node_req[2]
                self.phy_nodes.edges[self.sfc_map[action_idx], chosen_node]['bw'] -= self.backup_bw_req[0]


        else:

            self.wrong_backup_action = -10

    ###########################################################################################################

    def backup_removal(self, action_idx):

        if self.backup_map[action_idx] == -1:
            pass

        else:

            self.phy_nodes.nodes[self.backup_map[action_idx]]['CPU'] += self.backup_node_req[0]
            self.phy_nodes.nodes[self.backup_map[action_idx]]['storage'] += self.backup_node_req[1]
            self.phy_nodes.nodes[self.backup_map[action_idx]]['memory'] += self.backup_node_req[2]
            self.phy_nodes.edges[self.sfc_map[action_idx], self.backup_map[action_idx]]['bw'] += self.backup_bw_req[0]

            self.backup_map[action_idx] = -1
            self.backup_availability[action_idx] = 0

        return self.backup_map, self.phy_nodes.nodes.data(), self.phy_nodes.edges.data(), self.backup_availability

    #############################################################################################################
    def failure_protocol(self, action_idx):
        chosen_path = None
        reward = 0
        not_enough_resource = False
        i = 0
        done = False
        if self.backup_map[action_idx] != -1:
            self.wrong_fail_action = 0

            self.paths = list(nx.shortest_simple_paths(self.phy_nodes, self.sfc_map[action_idx],
                                                       self.backup_map[action_idx]))

            while not done:

                if i < len(self.paths) - 1:

                    done = all(self.phy_nodes.edges[self.paths[i][j], \
                                                    self.paths[i][j + 1]]["bw"] - self.backup_bw_req[1] >= 0 \
                               for j in range(len(self.paths[i]) - 1))
                    chosen_path = i

                else:
                    done = True
                    chosen_path = 0
                    not_enough_resource = True

                i += 1

            if not_enough_resource:

                self.borrowed_bw.append(self.backup_bw_req[1] - self.phy_nodes.edges[self.sfc_map[action_idx] \
                    , self.backup_map[action_idx]]['bw'])

                self.phy_nodes.edges[self.sfc_map[action_idx], self.backup_map[action_idx]]['bw'] = 0
                self.last_chosen_path.append(self.paths[0])
            else:
                self.borrowed_bw.append(0)

                self.last_chosen_path.append(self.paths[chosen_path])

                for j in range(len(self.paths[chosen_path]) - 1):
                    self.phy_nodes.edges[self.paths[chosen_path][j], \
                                         self.paths[chosen_path][j + 1]]['bw'] -= self.backup_bw_req[1]

            self.failure_protocol_map[action_idx] = 1



        else:
            # no backup
            self.wrong_fail_action = -50

    ############################################################################################################

    def failure_protocol_cleanup(self):
        for idx, val in np.ndenumerate(self.failure_protocol_map):
            if val == 1:
                self.backup_removal(idx)
        self.failure_protocol_map = np.zeros(self.sfc_map.shape)
        if len(self.last_chosen_path) != 0:
            for path in enumerate(self.last_chosen_path):
                for j in range(len(path[1]) - 1):
                    self.phy_nodes.edges[path[1][j], \
                                         path[1][j + 1]]['bw'] += \
                        (self.backup_bw_req[1] - self.borrowed_bw[path[0]])


        else:
            pass

        self.last_chosen_path = []
        self.failure_protocol_map = np.zeros(self.sfc_map.shape)

    def GetBW(self):
        d2 = nx.get_edge_attributes(self.phy_nodes, "bw")
        d1 = nx.get_edge_attributes(self.phy_nodes_init, "bw")
        bw_availablity = {k: d2[k] / d1[k] for k in d1.keys() & d2}

        return np.fromiter(bw_availablity.values(), dtype=np.float32)

    def GetBWRatio(self):
        d2 = self.backup_bw_req[0]
        d1 = nx.get_edge_attributes(self.phy_nodes_init, "bw")
        bw_req_ratio = {k: d2 / d1[k] for k in d1.keys()}

        return np.fromiter(bw_req_ratio.values(), dtype=np.float32)

    # def GetMaxBW(self):

    def GetCPURatio(self):
        d2 = nx.get_node_attributes(self.phy_nodes, "CPU")
        d1 = nx.get_node_attributes(self.phy_nodes_init, "CPU")
        CPU_availablity = {k: d2[k] / d1[k] for k in d1.keys() & d2}

        return np.fromiter(CPU_availablity.values(), dtype=np.float32)

    def GetCPUReqRatio(self):
        d2 = self.backup_node_req[0]
        d1 = nx.get_node_attributes(self.phy_nodes_init, "CPU")
        CPU_req_ratio = {k: d2 / d1[k] for k in d1.keys()}

        return np.fromiter(CPU_req_ratio.values(), dtype=np.float32)

    def GetStorage(self):
        d2 = nx.get_node_attributes(self.phy_nodes, "storage")
        d1 = nx.get_node_attributes(self.phy_nodes_init, "storage")
        storage_availablity = {k: d2[k] / d1[k] for k in d1.keys() & d2}

        return np.fromiter(storage_availablity.values(), dtype=np.float32)

    def GetStorageReqRatio(self):
        d2 = self.backup_node_req[1]
        d1 = nx.get_node_attributes(self.phy_nodes_init, "storage")
        storage_req_ratio = {k: d2 / d1[k] for k in d1.keys()}

        return np.fromiter(storage_req_ratio.values(), dtype=np.float32)

    def GetMemoryRatio(self):
        d2 = nx.get_node_attributes(self.phy_nodes, "memory")
        d1 = nx.get_node_attributes(self.phy_nodes_init, "memory")
        mem_availablity = {k: d2[k] / d1[k] for k in d1.keys() & d2}

        return np.fromiter(mem_availablity.values(), dtype=np.float32)

    def GetMemoryReqRatio(self):
        d2 = self.backup_node_req[2]
        d1 = nx.get_node_attributes(self.phy_nodes_init, "memory")
        memory_req_ratio = {k: d2 / d1[k] for k in d1.keys()}

        return np.fromiter(memory_req_ratio.values(), dtype=np.float32)

    def Getbackup_availability(self):

        for idx, i in np.ndenumerate(self.backup_map):
            self.backup_availability[idx] = 1 if i != -1 else 0

        return self.backup_availability.flatten()

    def Observation(self):
        observation = []
        #         observation.extend(self.GetBW().tolist())
        #         observation.extend(self.GetCPURatio().tolist())
        #         observation.extend(self.GetStorage().tolist())
        #         observation.extend(self.GetMemoryRatio().tolist())
        observation.extend(self.VNF_state.flatten().tolist())
        #         observation.extend(self.Getbackup_availability().tolist())

        return observation

    def GetObservationDimension(self):
        return len(self.Observation())

    def GetActionDimension(self):
        return self.sfc_map.shape

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self.__init__()
        self._state = self.Observation()
        self._episode_ended = False
        return ts.restart(np.array(self._state, dtype=np.float32))

    # def reward(self):
    def availability(self):
        for idx, node in np.ndenumerate(self.sfc_map):
            if self.backup_map[idx] != -1:
                self.availability_matrix[idx] = self.phy_nodes.nodes[node]['avail'] + \
                                                self.phy_nodes.nodes[self.backup_map[idx]]['avail'] - \
                                                (self.phy_nodes.nodes[self.backup_map[idx]]['avail'] *
                                                 self.phy_nodes.nodes[node]['avail'])
            else:
                self.availability_matrix[idx] = self.phy_nodes.nodes[node]['avail']

        return self.availability_matrix

    def availability_ratio(self):
        return np.prod(self.availability_matrix, axis=1, keepdims=True) - self.availability_req

    #     def reward(self):
    #         _reward = 0
    #         for idx, val in np.ndenumerate(self.VNF_state):
    #             if val == 0:
    #                 if self.backup_map[idx] != -1:
    #                     _reward -= 1
    #                 if self.failure_protocol_map[idx] != 0:
    #                     _reward -= 1

    #             elif val == 1:
    #                 if self.backup_map[idx] != -1:
    #                     _reward += 1
    #                 else:
    #                     _reward -= 1
    #                 if self.failure_protocol_map[idx] != 0:
    #                     _reward -= 1

    #             elif val == 2:
    #                 if self.failure_protocol_map[idx] == 1 and self.backup_map[idx] != -1 :
    #                     _reward += 100

    #                 elif self.failure_protocol_map[idx] == 1:
    #                     _reward += 5
    #                 else:
    #                     _reward -=10

    #         return _reward

    def _step(self, _action):
        action = _action.reshape(self.sfc_map.shape[0], self.sfc_map.shape[1])
        if self._episode_ended:
            return self.reset()

        _reward = 0

        self.failure_protocol_cleanup()

        for idx, val in np.ndenumerate(action):
            if -1<=val< -0.3:
                self.backup_placement(idx)
                if self.VNF_state[idx] == 1:
                    _reward += 1
                else:
                    _reward -= 1

            elif -0.3<=val < 0.3:
                if self.VNF_state[idx] == 2:

                    if self.backup_map[idx] != -1:
                        self.failure_protocol(idx)
                        _reward += 1e2
                    else:
                        self.failure_protocol_map[idx] = 1
                        _reward += 1
                    self.VNF_state[idx] = 0
                else:
                    _reward -= 1

            elif 0.3<= val <=1:
                self.backup_removal(idx)

                if self.VNF_state[idx] == 0:
                    _reward += 1
                else:
                    _reward -= 2

        #         self.availability()
        #         _reward += self.reward() + self.lost
        self.markov_trans()
        _reward += self.lost
        self._state = self.Observation()

        self.step_num += 1
        if self.step_num >= 100:
            self._episode_ended = True

        if self._episode_ended:
            #             if self.monitor_mode:
            #                 self.monitor()

            return ts.termination(np.array(self._state, dtype=np.float32), reward=_reward)

        else:

            return ts.transition(np.array(self._state, dtype=np.float32), reward=_reward, discount=0.99)

    def monitor(self):
        print("EPISODE:", "\n")
        print("\n", "SFC map", "\n", self.sfc_map)
        print("\n", "availability", "\n", self.availability_matrix)
        print("\n", "SFC_avail", "\n", np.prod(self.availability_matrix, axis=1, keepdims=True))
        print("\n", "backup_map", "\n", self.backup_map)
        print("\n", "failure protocol", "\n", self.failure_protocol_map)
        print("\n", "failure map", "\n", self.failure)

# ############################################# backup placement ####################################################