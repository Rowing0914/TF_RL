import tensorflow as tf


L2 = tf.keras.regularizers.l2(1e-2)
KERNEL_INIT = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)


class Nature_DQN(tf.keras.Model):
	def __init__(self, num_action):
		super(Nature_DQN, self).__init__()
		self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=8, strides=8, activation='relu')
		self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, activation='relu')
		self.conv3 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, activation='relu')
		self.flat = tf.keras.layers.Flatten()
		self.fc1 = tf.keras.layers.Dense(512, activation='relu')
		self.pred = tf.keras.layers.Dense(num_action, activation='linear')

	@tf.contrib.eager.defun(autograph=False)
	def call(self, inputs):
		x = self.conv1(inputs)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.flat(x)
		x = self.fc1(x)
		return self.pred(x)


class CartPole(tf.keras.Model):
	def __init__(self, num_action):
		super(CartPole, self).__init__()
		self.dense1 = tf.keras.layers.Dense(16, activation='relu')
		self.dense2 = tf.keras.layers.Dense(16, activation='relu')
		self.dense3 = tf.keras.layers.Dense(16, activation='relu')
		self.pred = tf.keras.layers.Dense(num_action, activation='linear')

	@tf.contrib.eager.defun(autograph=False)
	def call(self, inputs):
		x = self.dense1(inputs)
		x = self.dense2(x)
		x = self.dense3(x)
		return self.pred(x)


class Duelling_atari(tf.keras.Model):
	def __init__(self, num_action, duelling_type="avg"):
		super(Duelling_atari, self).__init__()
		self.duelling_type = duelling_type
		self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=8, strides=8, activation='relu')
		self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, activation='relu')
		self.conv3 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, activation='relu')
		self.flat = tf.keras.layers.Flatten()
		self.fc1 = tf.keras.layers.Dense(512, activation='relu')
		self.q_value = tf.keras.layers.Dense(num_action, activation='linear')
		self.v_value = tf.keras.layers.Dense(1, activation='linear')

	@tf.contrib.eager.defun(autograph=False)
	def call(self, inputs):
		x = self.conv1(inputs)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.flat(x)
		x = self.fc1(x)
		q_value = self.q_value(x)
		v_value = self.v_value(x)

		if self.duelling_type == "avg":
			# Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-Avg_a(A(s,a;theta)))
			output = tf.math.add(v_value, tf.math.subtract(q_value, tf.reduce_mean(q_value)))
		elif self.duelling_type == "max":
			# Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-max_a(A(s,a;theta)))
			output = tf.math.add(v_value, tf.math.subtract(q_value, tf.math.reduce_max(q_value)))
		elif self.duelling_type == "naive":
			# Q(s,a;theta) = V(s;theta) + A(s,a;theta)
			output = tf.math.add(v_value, q_value)
		else:
			output = 0 # defun does not accept the variable may not be intialised, so that temporarily initialise it
			assert False, "dueling_type must be one of {'avg','max','naive'}"
		return output


class Duelling_cartpole(tf.keras.Model):
	def __init__(self, num_action, duelling_type="avg"):
		super(Duelling_cartpole, self).__init__()
		self.duelling_type = duelling_type
		self.dense1 = tf.keras.layers.Dense(16, activation='relu')
		self.dense2 = tf.keras.layers.Dense(16, activation='relu')
		self.dense3 = tf.keras.layers.Dense(16, activation='relu')
		self.q_value = tf.keras.layers.Dense(num_action, activation='linear')
		self.v_value = tf.keras.layers.Dense(1, activation='linear')

	@tf.contrib.eager.defun(autograph=False)
	def call(self, inputs):
		x = self.dense1(inputs)
		x = self.dense2(x)
		x = self.dense3(x)
		q_value = self.q_value(x)
		v_value = self.v_value(x)

		if self.duelling_type == "avg":
			# Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-Avg_a(A(s,a;theta)))
			output = tf.math.add(v_value, tf.math.subtract(q_value, tf.reduce_mean(q_value)))
		elif self.duelling_type == "max":
			# Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-max_a(A(s,a;theta)))
			output = tf.math.add(v_value, tf.math.subtract(q_value, tf.math.reduce_max(q_value)))
		elif self.duelling_type == "naive":
			# Q(s,a;theta) = V(s;theta) + A(s,a;theta)
			output = tf.math.add(v_value, q_value)
		else:
			output = 0 # defun does not accept the variable may not be intialised, so that temporarily initialise it
			assert False, "dueling_type must be one of {'avg','max','naive'}"
		return output


class DDPG_Actor(tf.keras.Model):
	def __init__(self, num_action=1):
		super(DDPG_Actor, self).__init__()
		self.dense1 = tf.keras.layers.Dense(400, activation='relu', kernel_initializer=KERNEL_INIT)
		self.batch1 = tf.keras.layers.BatchNormalization()
		self.dense2 = tf.keras.layers.Dense(300, activation='relu', kernel_initializer=KERNEL_INIT)
		self.batch2 = tf.keras.layers.BatchNormalization()
		self.pred = tf.keras.layers.Dense(num_action, activation='tanh', kernel_initializer=KERNEL_INIT)

	@tf.contrib.eager.defun(autograph=False)
	def call(self, inputs):
		x = self.dense1(inputs)
		# x = self.batch1(x)
		x = self.dense2(x)
		x = self.batch2(x)
		pred = self.pred(x)
		return pred


class DDPG_Critic(tf.keras.Model):
	def __init__(self, output_shape):
		super(DDPG_Critic, self).__init__()
		self.dense1 = tf.keras.layers.Dense(400, activation='relu', kernel_regularizer=L2, bias_regularizer=L2, kernel_initializer=KERNEL_INIT)
		self.batch1 = tf.keras.layers.BatchNormalization()
		self.dense2 = tf.keras.layers.Dense(300, activation='relu', kernel_regularizer=L2, bias_regularizer=L2, kernel_initializer=KERNEL_INIT)
		self.batch2 = tf.keras.layers.BatchNormalization()
		self.pred = tf.keras.layers.Dense(output_shape, activation='linear', kernel_regularizer=L2, bias_regularizer=L2, kernel_initializer=KERNEL_INIT)

	@tf.contrib.eager.defun(autograph=False)
	def call(self, obs, act):
		x = self.dense1(obs)
		# x = self.batch1(x)
		x = self.dense2(tf.concat([x, act], axis=-1))
		x = self.batch2(x)
		pred = self.pred(x)
		return pred
