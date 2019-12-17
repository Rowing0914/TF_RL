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
L2 = tf.keras.regularizers.l2(1e-2)


class Actor(tf.keras.Model):
    def __init__(self, num_action=1):
        super(Actor, self).__init__()
        self.dense1 = tf.keras.layers.Dense(400, activation='relu', kernel_initializer=KERNEL_INIT)
        self.dense2 = tf.keras.layers.Dense(300, activation='relu', kernel_initializer=KERNEL_INIT)
        self.pred = tf.keras.layers.Dense(num_action, activation='tanh', kernel_initializer=KERNEL_INIT)

    @tf.function
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        pred = self.pred(x)
        return pred


class Critic(tf.keras.Model):
    def __init__(self, output_shape):
        super(Critic, self).__init__()
        self.dense1 = tf.keras.layers.Dense(400, activation='relu', kernel_regularizer=L2, bias_regularizer=L2,
                                            kernel_initializer=KERNEL_INIT)
        self.dense2 = tf.keras.layers.Dense(300, activation='relu', kernel_regularizer=L2, bias_regularizer=L2,
                                            kernel_initializer=KERNEL_INIT)
        self.pred = tf.keras.layers.Dense(output_shape, activation='linear', kernel_regularizer=L2, bias_regularizer=L2,
                                          kernel_initializer=KERNEL_INIT)

    @tf.function
    def call(self, obs, act):
        x = self.dense1(obs)
        x = self.dense2(tf.concat([x, act], axis=-1))
        pred = self.pred(x)
        return pred
