# original code
# https://github.com/dennybritz/reinforcement-learning/blob/master/PolicyGradient/CliffWalk%20REINFORCE%20with%20Baseline%20Solution.ipynb

# My contribution: I have re-programmed the code to use only Keras

import itertools
import numpy as np
import sys
import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Dense, Input
import collections

if "../" not in sys.path:
    sys.path.append("../")

from libs.envs.cliff_walking import CliffWalkingEnv

env = CliffWalkingEnv()

_action = 3


class PolicyEstimator():
    """
    Policy Function approximator. 
    """

    def __init__(self):
        # This is just table lookup estimator
        _in = Input(shape=(env.observation_space.n,))
        out = Dense(env.action_space.n, activation='softmax')(_in)
        self.model = Model(inputs=_in, outputs=out)

        def loss(y_true, y_pred):
            # using global action to access the picked action
            return - K.log(K.flatten(y_pred)[_action]) * y_true

        self.model.compile(loss=loss, optimizer="Adam")
        self.model.summary()

    def predict(self, state):
        return self.model.predict(keras.utils.to_categorical(state, int(env.observation_space.n)))[0]

    def update(self, state, target):
        return self.model.fit(keras.utils.to_categorical(state, int(env.observation_space.n)), target, verbose=0)


class ValueEstimator():
    """
    Value Function approximator. 
    """

    def __init__(self):
        _in = Input(shape=(env.observation_space.n,))
        out = Dense(1, activation='sigmoid')(_in)
        self.model = Model(inputs=_in, outputs=out)
        self.model.compile(loss='mse', optimizer='Adam', metrics=['accuracy'])
        self.model.summary()

    def predict(self, state):
        return self.model.predict(keras.utils.to_categorical(state, int(env.observation_space.n)))[0]

    def update(self, state, target):
        return self.model.fit(keras.utils.to_categorical(state, int(env.observation_space.n)), target, verbose=0)


def reinforce(env, estimator_policy, estimator_value, num_episodes, discount_factor=1.0):
    """
    REINFORCE (Monte Carlo Policy Gradient) Algorithm. Optimizes the policy
    function approximator using policy gradient.
    
    Args:
        env: OpenAI environment.
        estimator_policy: Policy Function to be optimized 
        estimator_value: Value function approximator, used as a baseline
        num_episodes: Number of episodes to run for
        discount_factor: Time-discount factor
    
    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    rewards_log = list()
    for i_episode in range(num_episodes):
        # Reset the environment and pick the fisrst action
        state = env.reset()

        episode = []
        rewards = 0

        # One step in the environment
        for t in itertools.count():

            # Take a step
            action_probs = estimator_policy.predict(np.array([state]))
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)

            # Keep track of the transition
            episode.append(Transition(
                state=state, action=action, reward=reward, next_state=next_state, done=done))

            # Update statistics
            rewards += reward

            # Print out which step we're on, useful for debugging.
            print("\rStep {} @ Episode {}/{} ({})".format(t, i_episode + 1, num_episodes, rewards), end="")
            # sys.stdout.flush()

            if done:
                rewards_log.append(rewards)
                break

            state = next_state

        # Go through the episode and make policy updates
        for t, transition in enumerate(episode):
            # The return after this timestep
            total_return = sum(discount_factor ** i * t.reward for i, t in enumerate(episode[t:]))
            # Calculate baseline/advantage
            baseline_value = estimator_value.predict(np.array([transition.state]))
            advantage = total_return - baseline_value
            # Update our value estimator
            estimator_value.update(np.array([transition.state]), np.array([total_return]))
            # Update our policy estimator
            global _action
            _action = transition.action
            estimator_policy.update(np.array([transition.state]), advantage)

    return rewards_log


policy_estimator = PolicyEstimator()
value_estimator = ValueEstimator()

# Note, due to randomness in the policy the number of episodes you need to learn a good
# policy may vary. ~2000-5000 seemed to work well for me.
stats = reinforce(env, policy_estimator, value_estimator, 2000, discount_factor=1.0)

import matplotlib.pyplot as plt

plt.plot(stats)
plt.show()
