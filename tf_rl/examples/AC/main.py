import gym
import itertools
import argparse
import numpy as np
import tensorflow as tf
from collections import deque

from tf_rl.common.eager_util import eager_setup

eager_setup()


class Actor(tf.keras.Model):
    def __init__(self, num_action):
        super(Actor, self).__init__()
        self.dense1 = tf.keras.layers.Dense(16, activation='relu')
        self.dense2 = tf.keras.layers.Dense(16, activation='relu')
        self.dense3 = tf.keras.layers.Dense(16, activation='relu')
        self.pred = tf.keras.layers.Dense(num_action, activation='softmax')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        pred = self.pred(x)
        return pred


class Critic(tf.keras.Model):
    def __init__(self):
        super(Critic, self).__init__()
        self.dense1 = tf.keras.layers.Dense(16, activation='relu')
        self.dense2 = tf.keras.layers.Dense(16, activation='relu')
        self.dense3 = tf.keras.layers.Dense(16, activation='relu')
        self.pred = tf.keras.layers.Dense(1, activation='linear')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        pred = self.pred(x)
        return pred


class ActorCritic(object):
    def __init__(self, actor, critic, num_action, gamma):
        self.gamma = gamma
        self.num_action = num_action
        self.actor = actor(num_action)
        self.critic = critic()
        self.actor_optimizer = tf.compat.v1.train.AdamOptimizer()
        self.critic_optimizer = tf.compat.v1.train.AdamOptimizer()

    def predict(self, state):
        return np.random.choice(np.arange(self.num_action),
                                p=self.actor(tf.convert_to_tensor(state[None, :], dtype=tf.float32)).numpy()[0])

    def update(self, state, action, reward, next_state, done):
        state = np.array(state[np.newaxis, ...]).astype(np.float32)
        action = np.array(action).astype(np.int32)
        reward = np.array(reward).astype(np.float32)
        next_state = np.array(next_state[np.newaxis, ...]).astype(np.float32)
        done = np.array(done).astype(np.float32)

        critic_loss, actor_loss = self.inner_update(state, action, reward, next_state, done)
        ts = tf.compat.v1.train.get_global_step()
        tf.summary.scalar("critic loss", critic_loss, ts)
        tf.summary.scalar("actor loss", actor_loss, ts)

    @tf.function
    def inner_update(self, state, action, reward, next_state, done):
        """
        Update Critic
        """

        with tf.GradientTape() as tape:
            state_value = self.critic(state)
            next_state_value = self.critic(next_state)
            td_target = reward + self.gamma * next_state_value * (1 - done)
            td_target = tf.stop_gradient(td_target)

            critic_loss = tf.losses.mean_squared_error(td_target, state_value)
            critic_loss = tf.reduce_mean(critic_loss)

        critic_grads = tape.gradient(critic_loss, self.critic.trainable_weights)

        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_weights))

        """
        Update Actor
        """

        with tf.GradientTape() as tape:
            action_probs = self.actor(tf.convert_to_tensor(state[None, :], dtype=tf.float32))

            actions_one_hot = tf.one_hot(action, self.num_action, 1.0, 0.0)
            action_probs = tf.math.reduce_sum(actions_one_hot * action_probs, axis=-1)
            advantage = tf.stop_gradient(td_target - state_value)

            actor_loss = tf.reduce_mean(-tf.math.log(action_probs) * advantage)

        actor_grads = tape.gradient(actor_loss, self.actor.trainable_weights)

        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_weights))
        return critic_loss, actor_loss


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="CartPole-v0", help="env name")
    parser.add_argument("--num_episodes", default=500, help="number of episodes")
    parser.add_argument("--gamma", default=0.99, help="discount factor")
    parser.add_argument("--log_dir", default="./logs/logs/AC", help="log dir name")
    params = parser.parse_args()

    env = gym.make(params.env_name)
    agent = ActorCritic(Actor, Critic, env.action_space.n, params.gamma)
    reward_buffer = deque(maxlen=10)
    writer = tf.summary.create_file_writer(logdir=params.log_dir)
    global_timestep = tf.compat.v1.train.create_global_step()

    with writer.as_default():
        for i in range(params.num_episodes):
            state = env.reset()
            total_reward = 0
            cnt_action = list()

            # generate an episode
            for t in itertools.count():
                action = agent.predict(state)
                next_state, reward, done, _ = env.step(action)

                # update the networks according to the current episode
                agent.update(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward
                cnt_action.append(action)
                global_timestep.assign_add(1)

                if done:
                    reward_buffer.append(total_reward)
                    tf.summary.scalar("score", np.mean(reward_buffer), step=global_timestep)

                    print(f"Episodes: {i} | Reward: {total_reward} | step: {t}")
                    total_reward = 0
                    break

            if np.mean(reward_buffer) > 190:
                break

        env.close()
