"""
simple tutorial
"""
import tensorflow as tf

x = tf.Variable(10.)
checkpoint = tf.train.Checkpoint()
checkpoint.mapped = {'x': x}
checkpoint_path = checkpoint.save('./tmp/')

x.assign(11.)  # Change the variable after saving.

# Restore values from the checkpoint
checkpoint = tf.train.Checkpoint()
checkpoint.mapped = {'x': x}
checkpoint.restore(checkpoint_path)

print(x.numpy())  # => 10.0