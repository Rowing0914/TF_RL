"""
This is the script for testing if we can revert the state of game
by directly accessing the internal state of Gym.Env class

"""

import gym


class MyWrapper_revertable(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.env = env.unwrapped
        self.env.seed(123)  # fix the randomness for reproducibility purpose

    def step(self, ac):
        next_state, reward, done, info = self.env.step(ac)
        if done:
            reward = -1.0  # reward at a terminal state
        return next_state, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset()

    def get_state(self):
        return self.env.state

    def set_state(self, state):
        self.env.state = state


# env = gym.make("MountainCarContinuous-v0")
# env = gym.make("MountainCar-v0")
env = MyWrapper(gym.make("MountainCar-v0"))

for i in range(5):
    state = env.reset()
    for t in range(100):
        env.render()
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        print(action, state, next_state, reward, done)

        if t > 10:
            print("SET ENV")
            env.set_state([0.5, 0.5])

        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
        state = next_state

env.close()
