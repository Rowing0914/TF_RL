import tensorflow as tf

class CartPoleModel(tf.keras.Model):
    def __init__(self, num_action):
        super(CartPoleModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(16, activation='relu')
        self.dense2 = tf.keras.layers.Dense(16, activation='relu')
        self.dense3 = tf.keras.layers.Dense(16, activation='relu')
        self.pred = tf.keras.layers.Dense(num_action, activation='linear')

    @tf.function
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.pred(x)


class Nature_DQN(tf.keras.Model):
    def __init__(self, num_action):
        super(Nature_DQN, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=8, strides=8, activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, activation='relu')
        self.flat = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(512, activation='relu')
        self.pred = tf.keras.layers.Dense(num_action, activation='linear')

    @tf.function
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flat(x)
        x = self.fc1(x)
        return self.pred(x)