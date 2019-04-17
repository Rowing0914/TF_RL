import gym
import numpy as np
import tensorflow as tf
from collections import deque
from tf_rl.common.wrappers import MyWrapper
from examples.params import Parameters
from tf_rl.common.memory import ReplayBuffer
from tf_rl.common.policy import EpsilonGreedyPolicy
from tf_rl.common.utils import AnnealingSchedule

class DQN:
    """
    On policy DQN

    """
    def __init__(self, num_action, scope):
        self.num_action = num_action
        self.scope = scope
        with tf.variable_scope(scope):
            self.state = tf.placeholder(shape=[None, 4], dtype=tf.float32, name="X")
            self.Y = tf.placeholder(shape=[None], dtype=tf.float32, name="Y")
            self.action = tf.placeholder(shape=[None], dtype=tf.int32, name="action")
            x = tf.keras.layers.Dense(16, activation=tf.nn.relu)(self.state)
            x = tf.keras.layers.Dense(16, activation=tf.nn.relu)(x)
            x = tf.keras.layers.Dense(16, activation=tf.nn.relu)(x)
            self.pred = tf.keras.layers.Dense(self.num_action, activation=tf.nn.relu, name="layer3")(x)
            self.actions_one_hot = tf.one_hot(self.action, self.num_action, 1.0, 0.0, name='action_one_hot')
            self.action_probs = tf.reduce_sum(self.actions_one_hot*self.pred, reduction_indices=-1)
            self.loss = tf.reduce_mean(tf.squared_difference(self.Y, self.action_probs))
            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

    def predict(self, sess, state):
        return sess.run(self.pred, feed_dict={self.state: state})

    def update(self, sess, state, action, target):
        feed_dict = {self.state: state, self.action: action, self.Y: target}
        loss, _ = sess.run([self.loss, self.optimizer], feed_dict=feed_dict)
        return loss

if __name__ == '__main__':
    env = MyWrapper(gym.make("CartPole-v0"))
    replay_buffer = ReplayBuffer(5000)
    params = Parameters(mode="CartPole")
    reward_buffer = deque(maxlen=params.reward_buffer_ep)
    Epsilon = AnnealingSchedule(start=params.epsilon_start, end=params.epsilon_end, decay_steps=params.decay_steps)
    policy = EpsilonGreedyPolicy(Epsilon_fn=Epsilon)

    agent = DQN(env.action_space.n, "agent")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(2000):
            state = env.reset()
            policy.index_episode = i

            total_reward = 0
            for t in range(210):
                # env.render()
                action = policy.select_action(sess, agent, state.reshape(params.state_reshape))
                next_state, reward, done, info = env.step(action)
                replay_buffer.add(state, action, reward, next_state, done)

                total_reward += reward

                if done:
                    print("Episode {0} finished after {1} timesteps".format(i, t + 1))

                    if i > 10:
                        states, actions, rewards, next_states, dones = replay_buffer.sample(params.batch_size)
                        next_Q = agent.predict(sess, next_states)
                        Y = rewards + params.gamma * np.max(next_Q, axis=1) * np.logical_not(dones)
                        loss = agent.update(sess, states, actions, Y)
                    break
                state = next_state


            reward_buffer.append(total_reward)

            if np.mean(reward_buffer) > params.goal:
                print("Game done!!")
                break

        env.close()

