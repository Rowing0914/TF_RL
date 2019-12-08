from graph_util.mujoco_parser import parse_mujoco_graph
from graph_util.graph_operator import GraphOperator
from scripts.dgl_samples.gcn import GCN

from collections import deque
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import gym

env_name = "WalkersHopperthree-v1"
env = gym.make(env_name)
node_info = parse_mujoco_graph(task_name=env_name)
graph_operator = GraphOperator(input_dict=node_info["input_dict"],
                               output_list=node_info["output_list"],
                               obs_shape=env.observation_space.shape[0])

graph_operator.get_all_attributes()
g = graph_operator._create_graph()
nx_G = g.to_networkx()

# nx.draw(nx_G)
# plt.show()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.gcn1 = GCN(1, 16, F.relu)
        self.gcn2 = GCN(16, 1, F.relu)

    def forward(self, g, features):
        x = self.gcn1(g, features)
        x = self.gcn2(g, x)
        return x

net = Net()
# print(net)
optimiser = th.optim.Adam(net.parameters(), lr=1e-3)
memory = deque(maxlen=100)

for ep in range(100):
    state = env.reset()
    for t in range(300):
        env.render()
        # action = env.action_space.sample()
        state = th.FloatTensor(state.reshape(11,1))
        action = net(g, state)
        action = graph_operator.readout_action(action)
        print(action)
        state, reward, done, info = env.step(action)
        memory.append((state, action, reward, done, info))

        if done:
            break
