import tensorflow as tf
import numpy as np
import os
import itertools
from common.visualise import plot_Q_values

"""

Utility functions 

"""

class AnnealingSchedule:
	"""
	Scheduling the gradually decreasign value, e.g., epsilon or beta params

	"""
	def __init__(self, start=1.0, end=0.1, decay_steps=500, decay_type="linear"):
		self.start       = start
		self.end = end
		self.decay_steps = decay_steps
		self.annealed_value = np.linspace(start, end, decay_steps)
		self.decay_type = decay_type

	def get_value(self, timestep):
		if self.decay_type == "linear":
			return self.annealed_value[min(timestep, self.decay_steps) - 1]
		elif self.decay_type == "curved":
			return max(self.end, min(1, 1.0 - np.log10((timestep + 1) / 25)))


def test(sess, agent, env, params):
	state = env.reset()

	xmax = agent.num_action
	ymax = 5

	print("\n ===== TEST STARTS =====  \n")

	for i in range(params.test_episodes):
		for t in itertools.count():
			env.render()
			q_values = sess.run(agent.pred, feed_dict={agent.state: state.reshape(params.state_reshape)})[0]
			action = np.argmax(q_values)
			plot_Q_values(q_values, xmax=xmax, ymax=ymax)
			obs, reward, done, _ = env.step(action)
			state = obs
			if done:
				print("Episode finished after {} timesteps".format(t + 1))
				break
	return


"""
Logging functions

.... maybe I don't use Tracker class...

"""


class Tracker:
	"""

	Tracking the data coming from the env and store them into a target file
	in Numpy Array for data visualisation purpose.

	Use store API to store the values, and since it does keep tracking the content of self.store
	whenever it gets full, this class adds those values to self.data and in the end, converts them into Numpy array
	and save them into a target file

	```python
	# instantiate
	tracker = Tracker(save_freq=100)

	# you can freely put the values in it
	tracker.store('state', state)
	```

	"""
	def __init__(self, play_data_file="../logs/data/log.npy", save_freq=1000):
		self.file = play_data_file
		self.save_freq = save_freq
		self.cnt = 0
		self.saved_cnt = 0
		self.data = list()
		self.value_names = [
			"state",
			"q_value",
			"action",
			"reward",
			"done",
			"loss",
		]
		self.storage_for_batch_names = ["loss"]

		self.storage_for_batch = {}
		self.storage = {}

		# refresh the content of target file
		try:
			os.remove(self.file)
		except: pass
		with open(self.file, "w"): pass

	def store(self, _key, value):
		"""
		Gets a value with a key and put them into a dictionary(self.storage)

		:param _key:
		:param value:
		:return:
		"""
		assert _key in self.value_names, "choose the value to store from {}".format(str(self.value_names))

		if len(self.storage) == len(self.value_names):
			self.add()
			self.storage = {}
		elif _key in self.storage_for_batch_names:
			if len(self.storage) == len(self.value_names):
				self.add()
				self.storage = {}
			self.storage_for_batch[_key] = value

		self.storage[_key] = value

	def add(self):
		"""
		We store data for visualising them later on!

		"""
		if self.cnt == self.save_freq:
			self._save_file()
		else:
			self.data.append(list(self.storage.values()))
			self.cnt += 1

	def _save_file(self):
		"""
		Save data into a file by Numpy array
		:return:
		"""
		print("WE SAVE THE PLAY DATA INTO {}".format(self.file))
		self.saved_cnt += 1
		try:
			prev_data = np.load(self.file)
		except:
			prev_data = np.zeros(len(self.storage))

		prev_data = np.vstack([prev_data, np.array(self.data)])
		self.cnt = 0
		self.data = list()

		np.save(self.file, prev_data)
		del prev_data



def logging(time_step, max_steps, current_episode, exec_time, reward, loss, epsilon, cnt_action):
	"""
	Logging function

	:param time_step:
	:param max_steps:
	:param current_episode:
	:param exec_time:
	:param reward:
	:param loss:
	:param cnt_action:
	:return:
	"""
	cnt_actions = dict((x, cnt_action.count(x)) for x in set(cnt_action))
	print("{0}/{1}: episode: {2}, duration: {3:.3f}s, episode reward: {4}, loss: {5:.6f}, epsilon: {6:.6f}, taken actions: {7}".format(
		time_step, max_steps, current_episode, exec_time, reward, loss, epsilon, cnt_actions
	))


"""

Update methods of a target model based on a source model 

"""

def sync_main_target(sess, main, target):
	"""
	Synchronise the models

	:param main:
	:param target:
	:return:
	"""
	e1_params = [t for t in tf.trainable_variables() if t.name.startswith(main.scope)]
	e1_params = sorted(e1_params, key=lambda v: v.name)
	e2_params = [t for t in tf.trainable_variables() if t.name.startswith(target.scope)]
	e2_params = sorted(e2_params, key=lambda v: v.name)

	update_ops = []
	for e1_v, e2_v in zip(e1_params, e2_params):
		op = e2_v.assign(e1_v)
		update_ops.append(op)
		
	sess.run(update_ops)


def soft_target_model_update(sess, main, target, tau=1e-2):
	"""
	Soft update model parameters.
	θ_target = τ*θ_local + (1 - τ)*θ_target

	:param main:
	:param target:
	:param tau:
	:return:
	"""
	e1_params = [t for t in tf.trainable_variables() if t.name.startswith(main.scope)]
	e1_params = sorted(e1_params, key=lambda v: v.name)
	e2_params = [t for t in tf.trainable_variables() if t.name.startswith(target.scope)]
	e2_params = sorted(e2_params, key=lambda v: v.name)

	update_ops = []
	for e1_v, e2_v in zip(e1_params, e2_params):

		# θ_target = τ*θ_local + (1 - τ)*θ_target
		op = e2_v.assign(tau*e1_v + (1 - tau)*e2_v)
		update_ops.append(op)

	sess.run(update_ops)



"""

Loss functions 

"""


def huber_loss(x, delta=1.0):
	"""
	Reference: https://en.wikipedia.org/wiki/Huber_loss
	"""
	return tf.where(
		tf.abs(x) < delta,
		tf.square(x) * 0.5,
		delta * (tf.abs(x) - 0.5 * delta)
	)


def ClipIfNotNone(grad, _min, _max):
	"""
	Reference: https://stackoverflow.com/a/39295309
	:param grad:
	:return:
	"""
	if grad is None:
		return grad
	return tf.clip_by_value(grad, _min, _max)
