# https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.python.eager import profiler

# Context manager APIs
with profiler.Profiler('./tmp'):
  # do your training here
  mnist = tf.keras.datasets.mnist

  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  x_train, x_test = x_train / 255.0, x_test / 255.0
  model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
  ])

  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  model.fit(x_train, y_train, epochs=5)

  model.evaluate(x_test, y_test, verbose=2)

# # Function APIs
# tf.python.eager.profiler.start()
# # do your training here
# profiler_result = tf.python.eager.profiler.stop()
# tf.python.eager.profiler.save('tmp', profiler_result)