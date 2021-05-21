import numpy as np
import networkx as nx
from random import sample


class SFC:
    def __init__(self, num_node=5, num_sfc=3, num_vnf=3):
        self.num_node = num_node
        self.num_sfc = num_sfc
        self.num_vnf = num_vnf + 2
        self.num_all_nodes = num_node + 2
        self.availability_matrix = np.zeros((num_sfc, num_vnf))

    ################################### physical network graph generator #####################################
    def physical_node(self):

        self.phy_graph = nx.complete_graph(self.num_node)
        for (i) in self.phy_graph.nodes():
            self.phy_graph.nodes[i]['CPU'] = np.random.uniform(low=250, high=400)
            self.phy_graph.nodes[i]['storage'] = np.random.uniform(low=350, high=570)
            self.phy_graph.nodes[i]['memory'] = np.random.uniform(low=500, high=850)
            self.phy_graph.nodes[i]['avail'] = np.random.uniform(low=0.95, high=0.99999)

        for (u, v) in self.phy_graph.edges():
            self.phy_graph.edges[u, v]["bw"] = np.random.uniform(low=150, high=250)

        #         nx.draw(phy_graph, with_labels = True)
        return self.phy_graph

    ############################### generating randon SFC from physical graph #################################
    def random_sfc(self):
        phy_graph = nx.complete_graph(self.num_all_nodes)
        sg = list(nx.all_simple_paths(phy_graph, source=self.num_all_nodes - 2, target=self.num_all_nodes - 1))
        _r = []

        for i in range(len(sg)):
            if len(sg[i]) == self.num_vnf:
                _r.append(sg[i])
        rand_sfc = sample(_r, self.num_sfc)
        sfc = np.array(rand_sfc)
        sfc_mapping = np.delete(sfc, obj=[0, self.num_vnf - 1], axis=1)
        tr_violation = np.zeros((sfc_mapping.shape[0], sfc_mapping.shape[1], 3))
        availability_req = np.random.uniform(high=0.99999999, low=0.98, size=(self.num_sfc, 1))
        for idx, node in np.ndenumerate(sfc_mapping):
            self.availability_matrix[idx] = self.phy_graph.nodes[node]['avail']

        return sfc_mapping, tr_violation, availability_req, self.availability_matrix
