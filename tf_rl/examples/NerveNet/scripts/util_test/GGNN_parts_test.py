"""
this is the test script of some minor components to
deeply understand their behaviour
"""

import gym, numpy as np
import tensorflow as tf
import graph_util.gnn_util as gnn_util
import graph_util.graph_data_util as graph_data_util
from graph_util.mujoco_parser import parse_mujoco_graph
from config.config import get_config
from environments import register

args = get_config()

# default value
_input_feat_dim = args.gnn_input_feat_dim

node_info = parse_mujoco_graph(task_name=args.task)
env = gym.make(args.task)

"""
=== node_info ===
dict(tree,
     relation_matrix
     node_type_dict,
     output_type_dict,
     input_dict,
     output_list,
     debug_info,
     node_parameters,
     para_size_dict,
     num_nodes)
"""

# step 2: check for ob size for each node type, construct the node dict
node_info = gnn_util.construct_ob_size_dict(node_info, _input_feat_dim)

# step 3: get the inverse node offsets (used to construct gather idx)
node_info = gnn_util.get_inverse_type_offset(node_info, 'node')

# step 4: get the inverse node offsets (used to gather output idx)
node_info = gnn_util.get_inverse_type_offset(node_info, 'output')

# step 5: register existing edge and get the receive and send index
node_info = gnn_util.get_receive_send_idx(node_info)

# for key, item in node_info.items():
#     print("========")
#     print(key, item)

"""
==============================================================================
check how to construct the placeholders in GGNN class
it's been defined in gated_graph_policy_network.py's _prepare method
==============================================================================
"""

# step 1: build the input_obs and input_parameters
_input_obs = {
    node_type: tf.placeholder(tf.float32,
                              [None, node_info['ob_size_dict'][node_type]])
    for node_type in node_info['node_type_dict']
}

input_parameter_dtype = tf.int32 if 'noninput' in args.gnn_embedding_option \
                                 else tf.float32
_input_parameters = {
    node_type: tf.placeholder(input_parameter_dtype,
                              [None, node_info['para_size_dict'][node_type]])
    for node_type in node_info['node_type_dict']
}
print("\n=== Step 1 ===")
print(_input_obs)
print(_input_parameters)

# step 2: the receive and send index
_receive_idx = tf.placeholder(
    tf.int32, shape=(None), name='receive_idx'
)
_send_idx = {
    edge_type: tf.placeholder(tf.int32, shape=(None),
                              name='send_idx_{}'.format(edge_type))
    for edge_type in node_info['edge_type_list']
}
print("\n=== Step 2 ===")
print(_receive_idx)
print(_send_idx)

# step 3: the node type index and inverse node type index
_node_type_idx = {
    node_type: tf.placeholder(tf.int32, shape=(None),
                              name='node_type_idx_{}'.format(node_type))
    for node_type in node_info['node_type_dict']
}
_inverse_node_type_idx = tf.placeholder(
    tf.int32, shape=(None), name='inverse_node_type_idx'
)
print("\n=== Step 3 ===")
print(_node_type_idx)
print(_inverse_node_type_idx)

# step 4: the output node index
_output_type_idx = {
    output_type: tf.placeholder(tf.int32, shape=(None),
                                name='output_type_idx_{}'.format(output_type))
    for output_type in node_info['output_type_dict']
}

_inverse_output_type_idx = tf.placeholder(
    tf.int32, shape=(None), name='inverse_output_type_idx'
)
print("\n=== Step 4 ===")
print(_output_type_idx)
print(_inverse_output_type_idx)

# step 5: batch_size
_batch_size_int = tf.placeholder(
    tf.int32, shape=(), name='batch_size_int'
)
print("\n=== Step 5 ===")
print(_batch_size_int)

"""
==============================================================================
check how to construct the feeddict
it's been defined in agent.py's prepared_policy_network_feeddict method!
==============================================================================
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


# the node information
# construct the graph input feed dict
# in this case, we need to get the receive_idx, send_idx,
# node_idx, inverse_node_idx ready. These index will be helpful
# to telling the network how to pass and update the information
state = env.reset()
obs, reward, done, info = env.step(env.action_space.sample())
obs_n = np.expand_dims(obs, 0)
print(state.shape, obs_n.shape)

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

feed_dict = {
    batch_size_int_placeholder: int(last_batch_size),
    receive_idx_placeholder: receive_idx,
    inverse_node_type_idx_placeholder: inverse_node_type_idx,
    inverse_output_type_idx_placeholder: inverse_output_type_idx
}

# as in agent.py's fetch_policy_info method L:79 shows
# we get _input_obs from gated_graph_policy_network
# so that I manually instantiate those variables
graph_obs_placeholder = _input_obs
graph_parameters_placeholder = _input_parameters

# append the input obs and parameters
for i_node_type in node_info['node_type_dict']:
    feed_dict[graph_obs_placeholder[i_node_type]] = graph_obs[i_node_type]
    feed_dict[graph_parameters_placeholder[i_node_type]] = \
        graph_parameters[i_node_type]

# append the send idx
for i_edge in node_info['edge_type_list']:
    feed_dict[send_idx_placeholder[i_edge]] = send_idx[i_edge]

# append the node type idx
for i_node_type in node_info['node_type_dict']:
    feed_dict[node_type_idx_placeholder[i_node_type]] = \
        node_type_idx[i_node_type]

# append the output type idx
for i_output_type in node_info['output_type_dict']:
    feed_dict[output_type_idx_placeholder[i_output_type]] = \
        output_type_idx[i_output_type]

for key, item in feed_dict.items():
    print("========")
    print(key, item)

for node_type in node_info["node_type_dict"]:
    print(node_type)
    print([args.gnn_input_feat_dim/2, node_info["para_size_dict"][node_type]])
    print([args.gnn_input_feat_dim/2, node_info["ob_size_dict"][node_type]])

for edge_type in node_info["edge_type_list"]:
    print(edge_type, args.network_shape + [args.gnn_node_hidden_dim]+[args.gnn_node_hidden_dim])
