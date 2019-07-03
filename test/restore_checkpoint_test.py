import tensorflow as tf
from tf_rl.agents.DQN import DQN
from tf_rl.common.params import Parameters
from tf_rl.common.policy import TestPolicy
import gym

tf.enable_eager_execution()
tf.random.set_random_seed(123)


class Model(tf.keras.Model):
    def __init__(self, env_type, num_action):
        super(Model, self).__init__()
        self.env_type = env_type
        if self.env_type == "CartPole":
            self.dense1 = tf.keras.layers.Dense(16, activation='relu')
            self.dense2 = tf.keras.layers.Dense(16, activation='relu')
            self.dense3 = tf.keras.layers.Dense(16, activation='relu')
            self.pred = tf.keras.layers.Dense(num_action, activation='linear')
        elif self.env_type == "Atari":
            self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=8, strides=8, activation='relu')
            self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, activation='relu')
            self.conv3 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, activation='relu')
            self.flat = tf.keras.layers.Flatten()
            self.fc1 = tf.keras.layers.Dense(512, activation='relu')
            self.pred = tf.keras.layers.Dense(num_action, activation='linear')

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


env = gym.make("CartPole-v0")
params = Parameters(algo="DQN", mode="CartPole")
expert = DQN("CartPole", Model, Model, env.action_space.n, params)
expert_policy = TestPolicy()
expert.check_point.restore(expert.manager.latest_checkpoint)
print(tf.train.get_global_step().numpy())
print("Restore the model from disk")
