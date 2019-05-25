import tensorflow as tf
import numpy as np
import os, datetime, itertools
from tf_rl.common.visualise import plot_Q_values

"""

Utility functions 

"""

class AnnealingSchedule:
	"""
	Scheduling the gradually decreasing value, e.g., epsilon or beta params

	"""
	def __init__(self, start=1.0, end=0.1, decay_steps=500, decay_type="linear"):
		self.start = start
		self.end = end
		self.decay_steps = decay_steps
		self.annealed_value = np.linspace(start, end, decay_steps)
		self.decay_type = decay_type

	def old_get_value(self, timestep):
		"""
		Deprecated

		:param timestep:
		:return:
		"""
		if self.decay_type == "linear":
			return self.annealed_value[min(timestep, self.decay_steps) - 1]
		# don't use this!!
		elif self.decay_type == "curved":
			if timestep < self.decay_steps:
				return self.start * 0.9 ** (timestep / self.decay_steps)
			else:
				return self.end

	def get_value(self):
		timestep = tf.train.get_or_create_global_step() # we are maintaining the global-step in train.py so it is accessible
		if self.decay_type == "linear":
			return self.annealed_value[min(timestep.numpy(), self.decay_steps) - 1]
		# don't use this!!
		elif self.decay_type == "curved":
			if timestep.numpy() < self.decay_steps:
				return self.start * 0.9 ** (timestep.numpy() / self.decay_steps)
			else:
				return self.end



def test(sess, agent, env, params):
	xmax = agent.num_action
	ymax = 3

	print("\n ===== TEST STARTS: {0} Episodes =====  \n".format(params.test_episodes))

	for i in range(params.test_episodes):
		state = env.reset()
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


class logger:
	def __init__(self, params):
		self.params = params
		self.prev_update_step = 0

	def logging(self, time_step, current_episode, exec_time, reward_buffer, loss, epsilon, cnt_action):
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
		# remaing_time_step/exec_time_for_one_step
		remaining_time = str(datetime.timedelta(seconds=(self.params.num_frames - time_step)*exec_time/(time_step - self.prev_update_step)))
		print("{0}/{1}: Ep: {2}({3:.3f}s), Remaining: {4}, (R) GOAL: {5}, {6} Ep => [MEAN: {7}, MAX: {8}], (last ep) Loss: {9:.6f}, Eps: {10:.6f}, Act: {11}".format(
			time_step, self.params.num_frames, current_episode, exec_time, remaining_time, self.params.goal, self.params.reward_buffer_ep, np.mean(reward_buffer), np.max(reward_buffer), loss, epsilon, cnt_actions
		))
		self.prev_update_step = time_step


"""

Update methods of a target model based on a source model 

"""

def sync_main_target(sess, target, source):
	"""
	Synchronise the models
	from Denny Britz's excellent RL repo
	https://github.com/dennybritz/reinforcement-learning/blob/master/DQN/Double%20DQN%20Solution.ipynb

	:param main:
	:param target:
	:return:
	"""
	source_params = [t for t in tf.trainable_variables() if t.name.startswith(source.scope)]
	source_params = sorted(source_params, key=lambda v: v.name)
	target_params = [t for t in tf.trainable_variables() if t.name.startswith(target.scope)]
	target_params = sorted(target_params, key=lambda v: v.name)

	update_ops = []
	for target_w, source_w in zip(target_params, source_params):
		op = target_w.assign(source_w)
		update_ops.append(op)
		
	sess.run(update_ops)


def soft_target_model_update(sess, target, source, tau=1e-2):
	"""
	Soft update model parameters.
	θ_target = τ*θ_local + (1 - τ)*θ_target

	:param main:
	:param target:
	:param tau:
	:return:
	"""
	source_params = [t for t in tf.trainable_variables() if t.name.startswith(source.scope)]
	source_params = sorted(source_params, key=lambda v: v.name)
	target_params = [t for t in tf.trainable_variables() if t.name.startswith(target.scope)]
	target_params = sorted(target_params, key=lambda v: v.name)

	update_ops = []
	for target_w, source_w in zip(target_params, source_params):

		# θ_target = τ*θ_local + (1 - τ)*θ_target
		op = target_w.assign(tau*source_w + (1 - tau)*target_w)
		update_ops.append(op)

	sess.run(update_ops)


def soft_target_model_update_eager(target, source, tau=1e-2):
	"""
	Soft update model parameters.
	θ_target = τ*θ_local + (1 - τ)*θ_target

	:param main:
	:param target:
	:param tau:
	:return:
	"""

	source_params = source.get_weights()
	target_params = target.get_weights()

	for target_w, source_w in zip(target_params, source_params):
		# θ_target = τ*θ_local + (1 - τ)*θ_target
		target_w = tau*target_w + (1 - tau)*source_w

	target.set_weights(target_params)


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




"""

Test Methods

"""


def test_Agent(agent, env, n_trial=1):
	"""
	Evaluate the trained agent!

	:return:
	"""
	all_rewards = list()
	print("=== Evaluation Mode ===")
	for ep in range(n_trial):
		state = env.reset()
		done = False
		episode_reward = 0
		while not done:
			# env.render()
			action = np.argmax(agent.predict(state))
			next_state, reward, done, _ = env.step(action)
			state = next_state
			episode_reward += reward
		all_rewards.append(episode_reward)
		tf.contrib.summary.scalar("Eval_Score over 250,000 time-step", episode_reward, step=agent.index_timestep)
		print("| Ep: {}/{} | Score: {} |".format(ep+1, n_trial, episode_reward))

	# if this is running on Google Colab, we would store the log/models to mounted MyDrive
	if agent.params.google_colab:
		os.rmdir(agent.params.log_dir_colab)
		os.rmdir(agent.params.model_dir_colab)
		os.system("cp -r {0} {1}".format(agent.params.log_dir, agent.params.log_dir_colab))
		os.system("cp -r {0} {1}".format(agent.params.model_dir, agent.params.model_dir_colab))

	if n_trial > 2:
		print("=== Evaluation Result ===")
		all_rewards = np.array([all_rewards])
		print("| Max: {} | Min: {} | STD: {} | MEAN: {} |".format(np.max(all_rewards), np.min(all_rewards), np.std(all_rewards), np.mean(all_rewards)))