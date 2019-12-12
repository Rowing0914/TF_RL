import gym
import numpy as np
import tensorflow as tf

from tf_rl.common.memory import ReplayBuffer
from tf_rl.examples.DQN.utils.policy import EpsilonGreedyPolicy_eager
from tf_rl.examples.DQN.utils.network import cartpole_net
from tf_rl.examples.DQN.utils.agent import dqn_agent


class Network(tf.Module):
    def __init__(self):
        self.dense1 = tf.keras.layers.Dense(16, activation="relu")
        self.dense2 = tf.keras.layers.Dense(16, activation="relu")
        self.dense3 = tf.keras.layers.Dense(1, activation="sigmoid")

    def __call__(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        out = tf.math.sign(x)
        return out


class Agent(object):
    def __init__(self):
        self.network = Network()
        self.optimizer = tf.compat.v1.train.AdamOptimizer()

    def update(self, states, expert_action):
        loss = self.inner_update(states, expert_action)
        return loss

    @tf.function
    def inner_update(self, states, expert_action):
        with tf.GradientTape() as tape:
            preds = self.network(states)
            loss = tf.losses.mean_squared_error(expert_action, preds)

        # get gradients
        grads = tape.gradient(loss, self.network.trainable_variables)

        # apply processed gradients to the network
        self.optimizer.apply_gradients(zip(grads, self.network.trainable_variables))
        return tf.math.reduce_mean(loss)


if __name__ == '__main__':
    env = gym.make("CartPole-v0")
    buffer = ReplayBuffer(size=1000)
    agent = Agent()
    expert = dqn_agent(model=cartpole_net,
                       policy=EpsilonGreedyPolicy_eager(num_action=env.action_space.n,
                                                        epsilon_fn=lambda: tf.constant(0.02)),
                       optimizer=tf.compat.v1.train.AdamOptimizer(),
                       loss_fn=tf.compat.v1.losses.huber_loss,
                       grad_clip_fn=lambda x: x,
                       num_action=env.action_space.n,
                       model_dir="./expert",
                       gamma=0.99,
                       obs_prc_fn=lambda x: x)
    reward_total = list()

    @tf.function
    def ask_expert(states):
        expert_action = expert.main_model(states)
        expert_action = tf.argmax(expert_action, axis=-1)
        return expert_action

    for epoch in range(100*5):
        state = env.reset()
        done = False
        reward_ep = 0
        while not done:
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            buffer.add(state, action, reward, next_state, done)
            state = next_state
            reward_ep += reward
        reward_total.append(reward_ep)

        losses = list()
        for grad_step in range(100):
            states, _, _, _, _ = buffer.sample(batch_size=32)
            expert_action = ask_expert(states)
            loss = agent.update(states, expert_action)
            losses.append(loss.numpy())

        print("Ep: {} Reward: {} MAR: {:.4f} Loss: {:.4f}".format(
            epoch, reward_ep, np.mean(reward_total), np.mean(losses)))

    env.close()
