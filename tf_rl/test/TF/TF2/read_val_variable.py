import tensorflow as tf

a = tf.Variable(tf.zeros((10, 10)))
print(a.read_value())
print(a[0])
print(a[2:5])