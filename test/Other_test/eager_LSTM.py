import numpy as np
import tensorflow as tf
from tf_rl.common.utils import eager_setup
from tensorflow.examples.tutorials.mnist import input_data

eager_setup()

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Training Parameters
learning_rate = 0.001
training_steps = 100
batch_size = 128
display_step = 200

# Network Parameters
num_input = 28  # MNIST data input (img shape: 28*28)
timesteps = 28  # timesteps
num_hidden = 128  # hidden layer num of features
num_classes = 10  # MNIST total classes (0-9 digits)


class LSTM(tf.keras.Model):
    def __init__(self, num_hidden, dim):
        super(LSTM, self).__init__(name="")
        self.dim = dim
        self.lstm_cell = tf.compat.v1.keras.layers.CuDNNLSTM(num_hidden, return_state=True, stateful=True)
        self.dense = tf.keras.layers.Dense(num_classes)

    def call(self, X):
        outputs = self.lstm_cell(X)
        outputs, state = outputs[0], outputs[1]
        return self.dense(outputs)

model = LSTM(num_hidden, num_input)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

test_len = 128
test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
test_label = mnist.test.labels[:test_len]

for step in range(1, training_steps + 1):
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    # Reshape data to get 28 seq of 28 elements
    batch_x = batch_x.reshape((batch_size, timesteps, num_input))

    with tf.GradientTape() as tape:
        logits = model(tf.convert_to_tensor(batch_x, np.float32))
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=batch_y))
        prediction = tf.nn.softmax(logits)
        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(batch_y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        print("Train Accuracy:{}".format(accuracy))

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    test_logits = model(tf.convert_to_tensor(test_data, np.float32))
    prediction = tf.nn.softmax(test_logits)
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(test_label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    print("Test Accuracy:{}".format(accuracy))
