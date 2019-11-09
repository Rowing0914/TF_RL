import tensorflow as tf
import functools

"""
[Note] Weight/Bias Initialisation
See how PyTorch one initialises the weights/biases in the network!

```python
# weight initialisation
nn.init.orthogonal_(layer.weight.data, gain=nn.init.calculate_gain('relu'))
# bias initialisation
nn.init.constant_(layer.bias.data, 0)
```

Originally, I was trying to follow this, BUT, Since Tensorflow's `tf.nn.relu` cannot return anything with input,
meaning we can't go like

```python
tf.keras.layers.Dense(kernel_initializer=tf.keras.initializers.Orthogonal(gain=tf.nn.relu)
``` 

So, I had to set the actual outcome of `nn.init.calculate_gain('relu')` returns `1.4142135623730951`.
"""

KERNEL_INIT = tf.keras.initializers.Orthogonal(gain=1.4142135623730951)
BIAS_INIT = tf.keras.initializers.zeros()

fast_converge_Dense = functools.partial(tf.keras.layers.Dense,
                                        kernel_initializer=KERNEL_INIT,
                                        bias_initializer=BIAS_INIT)
fast_converge_Conv2D = functools.partial(tf.keras.layers.Conv2D,
                                         kernel_initializer=KERNEL_INIT,
                                         bias_initializer=BIAS_INIT)


class cartpole_net(tf.keras.Model):
    def __init__(self, num_action):
        super(cartpole_net, self).__init__()
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


class atari_net(tf.keras.Model):
    def __init__(self, num_action, network_type="nature"):
        super(atari_net, self).__init__()
        if network_type == "nature":
            # Follows the original architecture
            self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=8, strides=8, activation='relu')
            self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, activation='relu')
            self.conv3 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, activation='relu')
            self.flat = tf.keras.layers.Flatten()
            self.fc1 = tf.keras.layers.Dense(256, activation='relu')
            self.pred = tf.keras.layers.Dense(num_action, activation='linear')
        elif network_type == "fast":
            # Follows the fast convergence architecture originated to PyTorch one!
            self.conv1 = fast_converge_Conv2D(filters=32, kernel_size=8, strides=8, activation='relu')
            self.conv2 = fast_converge_Conv2D(filters=64, kernel_size=4, strides=2, activation='relu')
            self.conv3 = fast_converge_Conv2D(filters=64, kernel_size=3, strides=1, activation='relu')
            self.flat = tf.keras.layers.Flatten()
            self.fc1 = fast_converge_Dense(units=256, activation='relu')
            self.pred = fast_converge_Dense(units=num_action, activation='linear')

    @tf.function
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flat(x)
        x = self.fc1(x)
        return self.pred(x)
