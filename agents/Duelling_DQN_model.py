import numpy as np
import tensorflow as tf
from common.utils import huber_loss, ClipIfNotNone


class Duelling_DQN:
	"""
	Duelling DQN Agent
	"""

	def __init__(self):
		"""
		define your model here!
		"""
		pass

	def predict(self, sess, state):
		"""
		predict q-values given a state

		:param sess:
		:param state:
		:return:
		"""
		return sess.run(self.pred, feed_dict={self.state: state})

	def update(self, sess, state, action, Y):
		feed_dict = {self.state: state, self.action: action, self.Y: Y}
		_, loss = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
		return loss


class Duelling_DQN_CartPole(Duelling_DQN):
	"""
	Duelling DQN Agent
	"""

	def __init__(self, scope, dueling_type, env, loss_fn="MSE"):
		self.scope = scope
		self.num_action = env.action_space.n
		with tf.variable_scope(scope):
			self.state = tf.placeholder(shape=[None, env.observation_space.shape[0]], dtype=tf.float32, name="X")
			self.Y = tf.placeholder(shape=[None], dtype=tf.float32, name="Y")
			self.action = tf.placeholder(shape=[None], dtype=tf.int32, name="action")

			fc1 = tf.keras.layers.Dense(16, activation=tf.nn.relu)(self.state)
			fc2 = tf.keras.layers.Dense(16, activation=tf.nn.relu)(fc1)
			self.pred = tf.keras.layers.Dense(self.num_action, activation=tf.nn.relu)(fc2)
			self.state_value = tf.keras.layers.Dense(1, activation=tf.nn.relu)(fc2)

			if dueling_type == "avg":
				# Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-Avg_a(A(s,a;theta)))
				self.output = tf.math.add(self.state_value, tf.math.subtract( self.pred, tf.reduce_mean(self.pred) ))
			elif dueling_type == "max":
				# Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-max_a(A(s,a;theta)))
				self.output = tf.math.add(self.state_value, tf.math.subtract( self.pred, tf.math.reduce_max(self.pred) ))
			elif dueling_type == "naive":
				# Q(s,a;theta) = V(s;theta) + A(s,a;theta)
				self.output = tf.math.add(self.state_value, self.pred)
			else:
				assert False, "dueling_type must be one of {'avg','max','naive'}"

			# indices of the executed actions
			idx_flattened = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action

			# passing [-1] to tf.reshape means flatten the array
			# using tf.gather, associate Q-values with the executed actions
			self.action_probs = tf.gather(tf.reshape(self.output, [-1]), idx_flattened)

			if loss_fn == "huber_loss":
				# use huber loss
				self.losses = tf.subtract(self.Y, self.action_probs)
				# self.loss = huber_loss(self.losses)
				self.loss = tf.reduce_mean(huber_loss(self.losses))
			elif loss_fn == "MSE":
				# use MSE
				self.losses = tf.squared_difference(self.Y, self.action_probs)
				self.loss = tf.reduce_mean(self.losses)
			else:
				assert False

			# you can choose whatever you want for the optimiser
			# self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
			self.optimizer = tf.train.AdamOptimizer()

			# to apply Gradient Clipping, we have to directly operate on the optimiser
			# check this: https://www.tensorflow.org/api_docs/python/tf/train/Optimizer#processing_gradients_before_applying_them
			self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
			self.clipped_grads_and_vars = [(ClipIfNotNone(grad, -1., 1.), var) for grad, var in self.grads_and_vars]
			self.train_op = self.optimizer.apply_gradients(self.clipped_grads_and_vars)



class Duelling_DQN_Atari(Duelling_DQN):
	"""
	Duelling DQN Agent
	"""

	def __init__(self, scope, dueling_type, env, loss_fn="MSE"):
		self.scope = scope
		self.num_action = env.action_space.n
		with tf.variable_scope(scope):
			self.state = tf.placeholder(shape=[None, 84, 84, 1], dtype=tf.float32, name="X")
			self.Y = tf.placeholder(shape=[None], dtype=tf.float32, name="Y")
			self.action = tf.placeholder(shape=[None], dtype=tf.int32, name="action")

			conv1 = tf.keras.layers.Conv2D(32, kernel_size=8, strides=8, activation=tf.nn.relu)(self.state)
			conv2 = tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, activation=tf.nn.relu)(conv1)
			conv3 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, activation=tf.nn.relu)(conv2)
			flat = tf.keras.layers.Flatten()(conv3)
			fc1 = tf.keras.layers.Dense(512, activation=tf.nn.relu)(flat)
			self.pred = tf.keras.layers.Dense(self.num_action, activation=tf.nn.relu)(fc1)
			self.state_value = tf.keras.layers.Dense(1, activation=tf.nn.relu)(fc1)

			if dueling_type == "avg":
				# Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-Avg_a(A(s,a;theta)))
				self.output = tf.math.add(self.state_value, tf.math.subtract( self.pred, tf.reduce_mean(self.pred) ))
			elif dueling_type == "max":
				# Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-max_a(A(s,a;theta)))
				self.output = tf.math.add(self.state_value, tf.math.subtract( self.pred, tf.math.reduce_max(self.pred) ))
			elif dueling_type == "naive":
				# Q(s,a;theta) = V(s;theta) + A(s,a;theta)
				self.output = tf.math.add(self.state_value, self.pred)
			else:
				assert False, "dueling_type must be one of {'avg','max','naive'}"

			# indices of the executed actions
			idx_flattened = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action

			# passing [-1] to tf.reshape means flatten the array
			# using tf.gather, associate Q-values with the executed actions
			self.action_probs = tf.gather(tf.reshape(self.output, [-1]), idx_flattened)

			if loss_fn == "huber_loss":
				# use huber loss
				self.losses = tf.subtract(self.Y, self.action_probs)
				# self.loss = huber_loss(self.losses)
				self.loss = tf.reduce_mean(huber_loss(self.losses))
			elif loss_fn == "MSE":
				# use MSE
				self.losses = tf.squared_difference(self.Y, self.action_probs)
				self.loss = tf.reduce_mean(self.losses)
			else:
				assert False

			# you can choose whatever you want for the optimiser
			# self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
			self.optimizer = tf.train.AdamOptimizer()

			# to apply Gradient Clipping, we have to directly operate on the optimiser
			# check this: https://www.tensorflow.org/api_docs/python/tf/train/Optimizer#processing_gradients_before_applying_them
			self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
			self.clipped_grads_and_vars = [(ClipIfNotNone(grad, -1., 1.), var) for grad, var in self.grads_and_vars]
			self.train_op = self.optimizer.apply_gradients(self.clipped_grads_and_vars)

