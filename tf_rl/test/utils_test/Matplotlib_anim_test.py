import numpy as np
from tf_rl.common.visualise import plot_Q_values

# for simple test

# for i in range(10):
# 	y = np.random.random(4)
# 	plot_Q_values(y, xmin=0, xmax=4, ymin=0, ymax=2)


# plotting a fake q_values during playing Cartpole
import gym
from tf_rl.common.wrappers import MyWrapper

env = MyWrapper(gym.make("CartPole-v0"))

for i in range(10):
    state = env.reset()
    for t in range(100):
        env.render()
        y = np.random.random(2)
        plot_Q_values(y, xmin=-1, xmax=2, ymin=0, ymax=2)
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        print(state, next_state, reward, done)
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
        state = next_state

env.close()
