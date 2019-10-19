import tensorflow as tf
from tf_rl.common.utils import eager_setup

eager_setup()
global_step = tf.train.get_or_create_global_step()

starter_learning_rate = 0.1
end_learning_rate = 0.01
decay_steps = 100
learning_rate = tf.train.polynomial_decay(starter_learning_rate, global_step,
                                          decay_steps, end_learning_rate,
                                          power=0.5)
print(learning_rate)

for _ in range(decay_steps):
    global_step.assign_add(1)
    print(learning_rate().numpy())
