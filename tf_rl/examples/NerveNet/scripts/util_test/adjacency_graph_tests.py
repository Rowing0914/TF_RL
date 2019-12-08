from graph_util.mujoco_parser import get_adjacency_matrix_Ant, get_adjacency_matrix

env_name = "CentipedeFour-v1"
# env_name = "CentipedeSix-v1"
A = get_adjacency_matrix(num_legs=8)
print(A.shape)
A = get_adjacency_matrix_Ant(visualise=True)
print(A.shape)
