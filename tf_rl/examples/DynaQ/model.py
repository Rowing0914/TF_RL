import numpy as np

from tf_rl.examples.Sutton_RL_Intro.libs.envs.grid_world import GridworldEnv


class Model(object):
    """ Tabular Env Dynamics Model """
    def __init__(self, num_state, num_action):
        self._transition = np.zeros(shape=(num_state, num_action), dtype=np.uint8)
        self._reward = np.zeros(shape=(num_state, num_action), dtype=np.float32)

    @property
    def transition_table(self):
        return self._transition

    @property
    def reward_table(self):
        return self._reward

    def update(self, state, action, reward, next_state):
        """ deterministically update the dynamics model """
        self._transition[state, action] = next_state
        self._reward[state, action] = reward

    def sample(self):
        """ randomly returns a previously visited state/action """
        state = np.random.choice(np.where(np.sum(self._transition, axis=-1) > 0)[0])
        action = np.random.choice(np.where(self._transition[state] > 0)[0])
        return state, action

    def step(self, state, action):
        """ transition to a next state given an action """
        next_state = self._transition[state, action]
        reward = self._reward[state, action]
        return next_state, reward


if __name__ == '__main__':
    """ Simple training script for the deterministic tabular dynamics model"""
    num_episode = num_steps = 20
    env = GridworldEnv()
    model = Model(num_state=env.observation_space.n, num_action=env.action_space.n)

    # Collect samples, Update the dynamics table
    for episode in range(num_episode):
        state = env.reset()
        for t in range(num_steps):
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            model.update(state=state, action=action, reward=reward, next_state=next_state)
            state = next_state

            if done: break

    env.close()
    print("Dynamics Table: ", model.transition_table)

    # Test the dynamics model
    state, action = model.sample()
    next_state, reward = model.step(state=state, action=action)

    print(state, action, reward, next_state)
