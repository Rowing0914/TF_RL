from graph_nets.utils_np import networkxs_to_graphs_tuple
from graph_util.mujoco_parser import _construct_kinematics_graph

g = _construct_kinematics_graph(num_legs=4)
networkxs_to_graphs_tuple([g])