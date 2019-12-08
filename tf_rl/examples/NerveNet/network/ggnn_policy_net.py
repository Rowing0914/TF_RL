import gym
import numpy as np
import tensorflow as tf
from tf_rl.common.utils import eager_setup

import graph_util.gnn_util as gnn_util
import graph_util.graph_data_util as graph_data_util
from config.config import get_config
from graph_util.mujoco_parser import parse_mujoco_graph

class MLP(tf.keras.Model):
    def __init__(self, dims, activations):
        super(MLP, self).__init__()
        self.num_layers = len(dims)-1
        self._summary = ""
        for i in range(self.num_layers):
            setattr(self, "layer_{}".format(i), tf.keras.layers.Dense(dims[i], activation=activations[i]))
            self._summary += "| layer: {} | dim: {} | activation: {} |\n".format(i+1, str(dims[i]), activations[i])

    def call(self, inputs):
        hidden = inputs
        for i in range(self.num_layers):
            hidden = getattr(self, "layer_{}".format(i))(hidden)
        return hidden

    def __str__(self):
        return self._summary

eager_setup()

args = get_config()
input_feat_dim = args.gnn_input_feat_dim

node_info = parse_mujoco_graph(task_name=args.task)
env = gym.make(args.task)

action_shape = env.action_space.shape[0]

"""
=== Build the node info by reading the XML
"""

# step 0-1: check for ob size for each node type, construct the node dict
node_info = gnn_util.construct_ob_size_dict(node_info, input_feat_dim)

# step 0-2: get the inverse node offsets (used to construct gather idx)
node_info = gnn_util.get_inverse_type_offset(node_info, 'node')

# step 0-3: get the inverse node offsets (used to gather output idx)
node_info = gnn_util.get_inverse_type_offset(node_info, 'output')

# step 0-4: register existing edge and get the receive and send index
node_info = gnn_util.get_receive_send_idx(node_info)

"""
=== Prepare variables
"""

# step 1: build the input_obs and input_parameters
_input_obs = {
    node_type: node_info['ob_size_dict'][node_type]
    for node_type in node_info['node_type_dict']
}

input_parameter_dtype = tf.int32 if 'noninput' in args.gnn_embedding_option else tf.float32
_input_parameters = {
    node_type: node_info['node_parameters'][node_type]
    for node_type in node_info['node_type_dict']
}
print("\n=== Step 1 ===")
print(_input_obs)
print(_input_parameters)

# step 2: the receive and send index
_receive_idx = None
_send_idx = {
    edge_type: edge_type
    for edge_type in node_info['edge_type_list']
}
print("\n=== Step 2 ===")
print(_receive_idx)
print(_send_idx)

# step 3: the node type index and inverse node type index
_node_type_idx = {
    node_type: node_type
    for node_type in node_info['node_type_dict']
}
_inverse_node_type_idx = None
print("\n=== Step 3 ===")
print(_node_type_idx)
print(_inverse_node_type_idx)

# step 4: the output node index
_output_type_idx = {
    output_type: output_type
    for output_type in node_info['output_type_dict']
}

_inverse_output_type_idx = None
print("\n=== Step 4 ===")
print(_output_type_idx)
print(_inverse_output_type_idx)

# step 5: batch_size
_batch_size_int = None
print("\n=== Step 5 ===")
print(_batch_size_int)

"""
=== Build the Networks
"""

# step 1-1: build the embedding network
# Tensor Shape: (None, para_size) -> (None, input_dim - ob_size)
MLP_embedding = {
    node_type: MLP(dims=[int(input_feat_dim / 2), node_info["para_size_dict"][node_type]], activations=["tanh"])
    for node_type in node_info["node_type_dict"]
    if node_info["ob_size_dict"][node_type] > 0
}

MLP_embedding.update({
    node_type: MLP([input_feat_dim, node_info["para_size_dict"][node_type]], activations=["tanh"])
    for node_type in node_info["node_type_dict"]
    if node_info["ob_size_dict"][node_type] == 0
})

for key, model in MLP_embedding.items():
    print(key, model)

# step 1-2: build the ob mapping network
MLP_ob_mapping = {
    node_type: MLP(dims=[int(input_feat_dim / 2), node_info["para_size_dict"][node_type]], activations=["tanh"])
    for node_type in node_info["node_type_dict"]
    if node_info["ob_size_dict"][node_type] > 0
}

for key, model in MLP_ob_mapping.items():
    print(key, model)

# step 1-3: build the mlp for the propagation network between nodes
MLP_prop_shape = args.network_shape + [args.gnn_node_hidden_dim, args.gnn_node_hidden_dim]
MLP_prop = {
    i_edge: MLP(MLP_prop_shape, activations=["tanh"] * (len(MLP_prop_shape) - 1))
    for i_edge in node_info["edge_type_list"]
}

for key, model in MLP_prop.items():
    print(key, model)

# step 1-4: build the node update function for each node type
Node_update = {
    i_node_type: tf.keras.layers.CuDNNGRU(args.gnn_node_hidden_dim)
    for i_node_type in node_info['node_type_dict']
}

# step 1-5: build the output network to readout the node features as the mu of each action(actuator)
MLP_out_shape = args.network_shape + [1] + [args.gnn_node_hidden_dim]
MLP_out = {
    output_type: MLP(MLP_out_shape, activations=["tanh"] * (len(MLP_out_shape) - 1))
    for output_type in node_info["output_type_dict"]
}

for key, model in MLP_out.items():
    print(key, model)

# step 1-6: build the log std for the actions
action_dist_logstd = tf.Variable((0.0 * np.random.randn(1, action_shape)).astype(np.float32))
print(action_dist_logstd)

"""
=== Connect the defined Networks
"""

# step 2-1 gather the input_feature from obs and node parameters
input_embedding = {
    node_type: MLP_embedding[node_type](_input_parameters[node_type].astype(np.float32))[-1]
    for node_type in node_info["node_type_dict"]
}

for key, value in input_embedding.items():
    print(key, value.shape)

"""
=== BEGIN ===
We need this part for converting the state into graph obs
"""


def get_gnn_idx_placeholder():
    """ this has been taken from gated_graph_policy_network.py's GGNN class """
    return _receive_idx, _send_idx, \
           _node_type_idx, _inverse_node_type_idx, \
           _output_type_idx, _inverse_output_type_idx, \
           _batch_size_int


# this part is coming from agent.py's fetch_policy_info method
receive_idx_placeholder, send_idx_placeholder, node_type_idx_placeholder, \
inverse_node_type_idx_placeholder, output_type_idx_placeholder, \
inverse_output_type_idx_placeholder, batch_size_int_placeholder = \
    get_gnn_idx_placeholder()

# in agent.py's gnn_parameter_initialization()
# we define those params below
receive_idx = None
send_idx = None
node_type_idx = None
inverse_node_type_idx = None
output_type_idx = None
inverse_output_type_idx = None
last_batch_size = -1

"""
=== END ===
"""

obs = env.reset()
obs_n = np.expand_dims(obs, 0)

"""
======================================================================
Note: At this point, we convert the state into the graph observation!!
======================================================================
"""
graph_obs, graph_parameters, receive_idx, send_idx, node_type_idx, \
inverse_node_type_idx, output_type_idx, \
inverse_output_type_idx, last_batch_size = \
    graph_data_util.construct_graph_input_feeddict(
        node_info,
        obs_n,
        receive_idx,
        send_idx,
        node_type_idx,
        inverse_node_type_idx,
        output_type_idx,
        inverse_output_type_idx,
        last_batch_size
    )

# this is the converted input
print(graph_obs)

"""
=== Fit the state at t to the Networks
"""

_ob_feat = {
    node_type: MLP_ob_mapping[node_type](graph_obs[node_type])[-1]
    for node_type in node_info["node_type_dict"]
    if node_info["ob_size_dict"][node_type] > 0
}
print("\n=== _ob_feat ===")
for key, value in _ob_feat.items():
    print(key, value.shape)

_input_feat = {
    node_type: tf.concat([input_embedding[node_type], tf.cast(_ob_feat[node_type], dtype=tf.float32)], axis=0)
    for node_type in node_info["node_type_dict"]
}
print("\n=== _input_feat ===")
for key, value in _input_feat.items():
    print(key, value.shape)

_input_node_hidden = _input_feat
_input_node_hidden = tf.concat([_input_node_hidden[node_type] for node_type in node_info["node_type_dict"]], axis=0)
print("\n=== prev: _input_node_hidden ===")
print(_input_node_hidden)
print(_input_node_hidden.shape)

print("inverse_node_type_idx: ", inverse_node_type_idx)
_input_node_hidden = tf.gather(_input_node_hidden, inverse_node_type_idx)
print("\n=== after: _input_node_hidden ===")
print(_input_node_hidden)
print("=================\n")

# step 2-2: unroll the propagation
_node_hidden = [None] * (args.gnn_num_prop_steps + 1)
_node_hidden[-1] = _input_node_hidden
_prop_msg = [None] * node_info["num_edge_type"]

for tt in range(args.gnn_num_prop_steps):
    ee = 0
    for i_edge_type in node_info["edge_type_list"]:
        node_active = tf.gather(_node_hidden[tt - 1], send_idx[i_edge_type])
        print(i_edge_type, node_active.shape)
        _prop_msg[ee] = MLP_prop[i_edge_type](tf.expand_dims(node_active, 0))[-1]
        ee += 1

    # aggregate messages
    concat_msg = tf.concat(_prop_msg, 0)
    print(concat_msg.shape, receive_idx, node_info["num_nodes"] * last_batch_size)

    """
    I couldn't figure out the error which occurs at this point.
    
    >> Error Message
    tensorflow.python.framework.errors_impl.InvalidArgumentError: data.shape = [320] 
    does not start with segment_ids.shape = [44] [Op:UnsortedSegmentSum]
    
    I think the dim of concat_msg and the receive_idx are supposed to be different ones to my current implementation..
    but don't know how to do
    """

    message = tf.unsorted_segment_sum(concat_msg, receive_idx, node_info["num_nodes"] * last_batch_size)
    denom_const = tf.unsorted_segment_sum(tf.ones_like(concat_msg), receive_idx,
                                          node_info["num_nodes"] * last_batch_size)
    message = tf.math.divide(message, (denom_const + tf.constant(1.0e-10)))
