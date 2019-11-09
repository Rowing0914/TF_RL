# refer to https://github.com/dennybritz/reinforcement-learning/blob/master/PolicyGradient/Continuous%20MountainCar%20Actor%20Critic%20Solution.ipynb
import collections
import gym
import itertools
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler

env = gym.envs.make("MountainCarContinuous-v0")

observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)

featurizer = sklearn.pipeline.FeatureUnion([
    ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
    ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
    ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
    ("rbf4", RBFSampler(gamma=0.5, n_components=100))
])
featurizer.fit(scaler.transform(observation_examples))


def featurize_state(state):
    scaled = scaler.transform([state])
    featurized = featurizer.transform(scaled)
    return featurized[0]


class PolicyEstimator:
    def __init__(self, learning_rate=0.01, scope="policy_estimator"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.float32, [400], "state")
            self.target = tf.placeholder(dtype=tf.float32, name="target")
            self.mu = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(self.state, 0),
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer)
            self.mu = tf.squeeze(self.mu)
            self.sigma = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(self.state, 0),
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer)
            self.sigma = tf.squeeze(self.sigma)
            self.sigma = tf.nn.softplus(self.sigma) + 1e-5
            self.normal_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)
            self.action = self.normal_dist._sample_n(1)
            self.action = tf.clip_by_value(self.action, env.action_space.low[0], env.action_space.high[0])
            self.loss = self.normal_dist.log_prob(self.action) * self.target
            self.loss -= 1e-1 * self.normal_dist.entropy()
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.train.get_global_step())

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        state = featurize_state(state)
        return sess.run(self.action, {self.state: state})

    def update(self, state, target, action, sess=None):
        sess = sess or tf.get_default_session()
        state = featurize_state(state)
        feed_dict = {self.state: state, self.target: target, self.action: action}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


class ValueEstimator:
    def __init__(self, learning_rate=0.1, scope="value_estimator"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.float32, [400], "state")
            self.target = tf.placeholder(dtype=tf.float32, name="target")
            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(self.state, 0),
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer)
            self.value_estimate = tf.squeeze(self.output_layer)
            self.loss = tf.squared_difference(self.value_estimate, self.target)
            self.optimizer = tf.train.AdamOptimizer(learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.train.get_global_step())

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        state = featurize_state(state)
        return sess.run(self.value_estimate, {self.state: state})

    def update(self, state, target, sess=None):
        sess = sess or tf.get_default_session()
        state = featurize_state(state)
        feed_dict = {self.state: state, self.target: target}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


def actor_critic(env, estimator_policy, estimator_value, num_episodes, discount_factor=1.0):
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    reward_log = list()

    for i_episode in range(num_episodes):
        state = env.reset()
        episode = []
        rewards = 0

        for i in itertools.count():
            action = estimator_policy.predict(state)
            next_state, reward, done, _ = env.step(action)
            rewards += reward

            value_next = estimator_value.predict(next_state)
            td_target = reward + discount_factor * value_next
            td_error = td_target - estimator_value.predict(state)

            estimator_value.update(state, td_target)
            estimator_policy.update(state, td_error, action)
            print("\rStep {} @ Episode {}/{} ({})".format(i, i_episode + 1, num_episodes, rewards), end="")

            if done:
                reward_log.append(rewards)
                break
    return reward_log


if __name__ == '__main__':
    tf.reset_default_graph()
    global_step = tf.Variable(0, name="global_step", trainable=False)
    policy_estimator = PolicyEstimator()
    value_estimator = ValueEstimator()

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        log = actor_critic(env, policy_estimator, value_estimator, 50, discount_factor=0.95)

    plt.plot(log)
    plt.show()
