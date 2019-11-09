# Algorithm: on page 333 of the Sutton's RL book

import gym
import itertools
import numpy as np
import time
from collections import deque
from tf_rl.common.wrappers import MyWrapper
from tf_rl.common.params import Parameters
from tf_rl.common.utils import logging
import tensorflow as tf

config = tf.ConfigProto(allow_soft_placement=True,
                        intra_op_parallelism_threads=1,
                        inter_op_parallelism_threads=1)
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)
tf.random.set_random_seed(123)


class Policy_Network(tf.keras.Model):
    """
    Produces the action probability distirbution!
    Not Q-values as in Q-learning or other value-based RL methods

    """

    def __init__(self, env_type, num_action):
        super(Policy_Network, self).__init__()
        self.env_type = env_type
        if self.env_type == "CartPole":
            self.dense1 = tf.keras.layers.Dense(16, activation='relu')
            self.dense2 = tf.keras.layers.Dense(16, activation='relu')
            self.dense3 = tf.keras.layers.Dense(16, activation='relu')
            self.pred = tf.keras.layers.Dense(num_action, activation='softmax')
        elif self.env_type == "Atari":
            self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=8, strides=8, activation='relu')
            self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, activation='relu')
            self.conv3 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, activation='relu')
            self.flat = tf.keras.layers.Flatten()
            self.fc1 = tf.keras.layers.Dense(512, activation='relu')
            self.pred = tf.keras.layers.Dense(num_action, activation='softmax')

    def call(self, inputs):
        if self.env_type == "CartPole":
            x = self.dense1(inputs)
            x = self.dense2(x)
            x = self.dense3(x)
            pred = self.pred(x)
            return pred
        elif self.env_type == "Atari":
            x = self.conv1(inputs)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.flat(x)
            x = self.fc1(x)
            pred = self.pred(x)
            return pred


class Value_Network(tf.keras.Model):
    """
    Produces the state-value

    """

    def __init__(self, env_type):
        super(Value_Network, self).__init__()
        self.env_type = env_type
        if self.env_type == "CartPole":
            self.dense1 = tf.keras.layers.Dense(16, activation='relu')
            self.dense2 = tf.keras.layers.Dense(16, activation='relu')
            self.dense3 = tf.keras.layers.Dense(16, activation='relu')
            self.pred = tf.keras.layers.Dense(1, activation='linear')
        elif self.env_type == "Atari":
            self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=8, strides=8, activation='relu')
            self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, activation='relu')
            self.conv3 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, activation='relu')
            self.flat = tf.keras.layers.Flatten()
            self.fc1 = tf.keras.layers.Dense(512, activation='relu')
            self.pred = tf.keras.layers.Dense(1, activation='linear')

    def call(self, inputs):
        if self.env_type == "CartPole":
            x = self.dense1(inputs)
            x = self.dense2(x)
            x = self.dense3(x)
            pred = self.pred(x)
            return pred
        elif self.env_type == "Atari":
            x = self.conv1(inputs)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.flat(x)
            x = self.fc1(x)
            pred = self.pred(x)
            return pred


class REINFORCE:
    def __init__(self, env_type, policy_net, value_net, num_action):
        self.env_type = env_type
        self.num_action = num_action
        self.policy_net = policy_net(env_type, num_action)
        self.value_net = value_net(env_type)
        self.policy_net_optimizer = tf.train.AdamOptimizer()
        self.value_net_optimizer = tf.train.AdamOptimizer()

    def predict(self, state):
        # we take an action according to the action distribution produced by policy network
        return np.random.choice(np.arange(self.num_action),
                                p=self.policy_net(tf.convert_to_tensor(state[None, :], dtype=tf.float32)).numpy()[0])

    def update(self, memory):
        for step, data in enumerate(memory):
            # after an episode, we update networks(Policy and Value)
            state, action, next_state, reward, done = data

            # calculate discounted G
            total_return = sum(params.gamma ** i * t[3] for i, t in enumerate(memory[step:]))

            """
            
            Update Value Network
            
            """

            with tf.GradientTape() as tape:
                # calculate an advantage
                state_value = self.value_net(tf.convert_to_tensor(state[None, :], dtype=tf.float32))
                advantage = total_return - state_value

                # MSE loss function: (1/N)*sum(Advantage - V(s))^2
                value_net_loss = tf.reduce_mean(tf.squared_difference(advantage, state_value))

            # get gradients
            value_net_grads = tape.gradient(value_net_loss, self.value_net.trainable_weights)

            # apply processed gradients to the network
            self.value_net_optimizer.apply_gradients(zip(value_net_grads, self.value_net.trainable_weights))

            """

            Update ValPolicy ue Network

            """

            with tf.GradientTape() as tape:
                # compute action probability distirbution
                action_probs = self.policy_net(tf.convert_to_tensor(state[None, :], dtype=tf.float32))

                # get the probability according to the taken action in an episode
                actions_one_hot = tf.one_hot(action, self.num_action, 1.0, 0.0)
                action_probs = tf.reduce_sum(actions_one_hot * action_probs, reduction_indices=-1)

                # loss for policy network: TD_error * log p(a|s)
                policy_net_loss = -tf.log(action_probs) * advantage

            # get gradients
            policy_net_grads = tape.gradient(policy_net_loss, self.policy_net.trainable_weights)

            # apply processed gradients to the network
            self.policy_net_optimizer.apply_gradients(zip(policy_net_grads, self.policy_net.trainable_weights))


# env_type = "CartPole"
env = MyWrapper(gym.make("CartPole-v0"))
agent = REINFORCE("CartPole", Policy_Network, Value_Network, num_action=env.action_space.n)
params = Parameters(algo="REINFORCE", mode="CartPole")
reward_buffer = deque(maxlen=params.reward_buffer_ep)
global_timestep = 0

for i in range(params.num_episodes):
    state = env.reset()
    memory = list()
    total_reward = 0
    cnt_action = list()
    start = time.time()

    # generate an episode
    for t in itertools.count():
        # env.render()
        action = agent.predict(state)
        next_state, reward, done, info = env.step(action)
        memory.append([state, action, next_state, reward, done])

        state = next_state
        total_reward += reward
        cnt_action.append(action)

        if done:
            # logging purpose
            reward_buffer.append(total_reward)

            logging(global_timestep, params.num_frames, i, time.time() - start, total_reward, 0, 0, cnt_action)
            total_reward = 0

            # update the networks according to the current episode
            agent.update(memory)
            break

    # stopping condition: if the agent has achieved the goal twice successively then we stop this!!
    if np.mean(reward_buffer) > params.goal:
        break

env.close()
