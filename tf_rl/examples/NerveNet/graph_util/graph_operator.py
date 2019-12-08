import operator
from functools import reduce
import numpy as np
import dgl

class GraphOperator:
    """
    this class deals with converts the obs into a graph by using the node info provided by "mujoco_parser.py"

    - sample input_dict: {0: [0, 1, 5, 6, 7], 1: [2, 8], 2: [3, 9], 3: [4, 10]}
    - sample output_list: [1, 2, 3, 4, 7, 8, 9, 10, 5, 6]

    """

    def __init__(self, input_dict, output_list, obs_shape):
        self._create_nodes_indices(input_dict)
        self._create_action_indices(output_list)
        self._define_send_receive(input_dict)
        self._create_edges(obs_shape)  # TODO: what values should be edges
        self._create_globals()

    def _create_globals(self):
        self.globals = np.random.randn(1)  # TODO: what value should be globals
        self.num_globals = len(self.globals)

    def _create_nodes_indices(self, input_dict):
        """ Given the input obs, we order it and treat as a feature/features of a node """
        temp_indices = [v for k, v in input_dict.items()]
        self._node_indices = reduce(operator.concat, temp_indices)
        self.num_nodes = len(self._node_indices)

    def _create_action_indices(self, output_list):
        """ Create the action indices to readout the action from processed nodes features """
        self._action_indices = [self._node_indices.index(v) for v in output_list]
        self.num_action = len(self._action_indices)

    def _create_edges(self, obs_shape):
        """ Define edges which express the relation of two nodes(sender and receiver) """
        self.edges = np.random.uniform(low=-0.3, high=0.3, size=(obs_shape, 1))
        self.num_edges = obs_shape

    def _define_send_receive(self, input_dict):
        """ Defines the relationship between senders and receivers on edges """
        self.senders, self.receivers = list(), list()
        for key, value in input_dict.items():
            for v in value:
                # sen -> rec on one edge
                # self.senders.append(key)
                # self.receivers.append(v)
                self.senders.append(v)
                self.receivers.append(key)

    def _create_graph(self):
        g = dgl.DGLGraph()
        g.add_nodes(self.num_nodes)
        g.add_edges(self.senders, self.receivers)
        return g

    def get_all_attributes(self):
        for key, value in self.__dict__.items():
            print(key, value)

    def readout_action(self, node_features):
        """ Convert the resulting graph into an action """
        return node_features.cpu().detach().numpy().flatten()[self._action_indices]

    # def obs_to_graph(self, obs):
    #     """ Convert obs into a graph for Deepmind's Graph_net
    #     output: input_graph which contains the all relations among nodes and edges
    #     """
    #     nodes = tf.compat.v1.gather(tf.compat.v1.cast(obs, dtype=tf.float32), self._node_indices)  # Ordering the obs
    #     nodes = tf.compat.v1.reshape(nodes, (self.num_nodes, 1))
    #
    #     data_dict = {
    #         "globals": self.globals,
    #         "nodes": nodes,
    #         "edges": self.edges,
    #         "receivers": self.receivers,
    #         "senders": self.senders
    #     }
    #     return data_dicts_to_graphs_tuple([data_dict])