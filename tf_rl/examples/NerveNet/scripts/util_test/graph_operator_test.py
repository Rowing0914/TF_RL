from graph_util.mujoco_parser import parse_mujoco_graph
from graph_util.graph_operator import GraphOperator
import networkx as nx
import matplotlib.pyplot as plt

node_info = parse_mujoco_graph(task_name="WalkersHopperone-v1")
graph_operator = GraphOperator(input_dict=node_info["input_dict"],
                               output_list=node_info["output_list"],
                               obs_shape=11)

graph_operator.get_all_attributes()
g = graph_operator._create_graph()
nx_G = g.to_networkx()
nx.draw(nx_G)
plt.show()
