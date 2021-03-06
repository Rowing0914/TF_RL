import numpy as np
import tensorflow as tf


class Agent:
    """
    boiler plate of an agent in general

    """

    def __init__(self):
        pass

    def predict(self, state):
        raise NotImplementedError

    def _select_action(self, state):
        raise NotImplementedError

    def update(self, states, actions, rewards, next_states, dones):
        states = np.array(states, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        actions = np.array(actions, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        return self._inner_update(states, actions, rewards, next_states, dones)

    def _inner_update(self, states, actions, rewards, next_states, dones):
        raise NotImplementedError


class Agent_atari(Agent):
    """
    boiler plate of an agent for Atari game

    """

    def __init__(self):
        pass

    def predict(self, state):
        state = np.expand_dims(state / 255., axis=0).astype(np.float32)
        action = self._select_action(tf.constant(state))
        return action.numpy()[0]

    def update(self, states, actions, rewards, next_states, dones):
        states = np.array(states, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        actions = np.array(actions, dtype=np.uint8)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        return self._inner_update(states, actions, rewards, next_states, dones)

    def _select_action(self, state):
        raise NotImplementedError

    def _inner_update(self, states, actions, rewards, next_states, dones):
        raise NotImplementedError


class Agent_cartpole(Agent):
    """
    boiler plate of an agent for Atari game

    """

    def __init__(self):
        pass

    def predict(self, state):
        state = np.expand_dims(state, axis=0).astype(np.float32)
        action = self._select_action(tf.constant(state))
        return action.numpy()[0]

    def _select_action(self, state):
        raise NotImplementedError

    def update(self, states, actions, rewards, next_states, dones):
        states = np.array(states, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        actions = np.array(actions, dtype=np.uint8)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        return self._inner_update(states, actions, rewards, next_states, dones)

    def _inner_update(self, states, actions, rewards, next_states, dones):
        raise NotImplementedError
