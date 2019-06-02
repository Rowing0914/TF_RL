import gym
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print("p: {}/{}".format(rank, size))

env = gym.make("CartPole-v1")
state = env.reset()
memory = list()

for t in range(10):
	action = env.action_space.sample()
	next_state, reward, done, info = env.step(action)
	memory.append((next_state, reward, done, info))
	# print(reward, next_state)
	state = next_state
	if done:
		break
env.close()

recvbuf = comm.gather(memory, root=0)

if rank == 0:
	print(len(recvbuf))
	print(recvbuf)

# from mpi4py import MPI
# import numpy as np
#
# comm = MPI.COMM_WORLD
# size = comm.Get_size()
# rank = comm.Get_rank()
#
# sendbuf = np.zeros(100, dtype='i') + rank
# recvbuf = None
#
# if rank == 0:
# 	recvbuf = np.empty([size, 100], dtype='i')
# comm.Gather(sendbuf, recvbuf, root=0)
#
# if rank == 0:
# 	print(recvbuf)
# 	for i in range(size):
# 		assert np.allclose(recvbuf[i, :], i)
