import gym
import numpy as np
from tf_rl.common.utils import eager_setup
import graph_util.gnn_util as gnn_util
from graph_util.mujoco_parser import parse_mujoco_graph
from config.config import get_config
from network.ggnn import GGNN
import environments.register

eager_setup()

args = get_config()
input_feat_dim = args.gnn_input_feat_dim

node_info = parse_mujoco_graph(task_name=args.task)
node_info = gnn_util.add_node_info(node_info, input_feat_dim=input_feat_dim)
ggnn = GGNN(state_dim=15, node_info=node_info)

env = gym.make(args.task)
action_shape = env.action_space.shape[0]

state = env.reset()

for i in range(10):
    # env.render()
    action = ggnn(state[np.newaxis, ...])
    state, reward, done, info = env.step(action)