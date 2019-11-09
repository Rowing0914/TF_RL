import gym
from tf_rl.common.memory import ReplayBuffer
size = 100000
env = gym.make("CartPole-v0")
memory = ReplayBuffer(size=size, traj_dir="./traj/")

state = env.reset()
action = env.action_space.sample()
next_state, reward, done, info = env.step(action)
env.close()

for _ in range(size):
    memory.add(state, action, reward, next_state, done)
print(len(memory))
memory.save()

del memory
memory = ReplayBuffer(size=size, recover_data=True, traj_dir="./traj/")
print(len(memory))