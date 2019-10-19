# https://www.tensorflow.org/guide/checkpoint#list_and_dictionary_tracking
import tensorflow as tf

save = tf.train.Checkpoint()
save.listed = [tf.Variable(1.)]
save.listed.append(tf.Variable(2.))
save.mapped = {'one': save.listed[0]}
save.mapped['two'] = save.listed[1]
save_path = save.save('./tmp/tf_list_example')

restore = tf.train.Checkpoint()
v2 = tf.Variable(0.)
assert 0. == v2.numpy()  # Not restored yet
restore.mapped = {'two': v2}
restore.restore(save_path)
assert 2. == v2.numpy()