import gin
import tensorflow as tf
import tf_rl.common.gin_configurables

@gin.configurable
class Model(object):
    def __init__(self, optimizer=tf.optimizers.RMSprop, activation_fn=tf.nn.tanh):
        self._optimizer = optimizer()
        self._activation_fn = activation_fn

if __name__ == '__main__':
    gin.parse_config_file("./config.gin")
    model = Model()
    print(model._optimizer)