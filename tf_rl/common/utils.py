import tensorflow as tf
import numpy as np
import os, datetime, itertools, shutil
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
		timestep = tf.train.get_or_create_global_step()  # we are maintaining the global-step in train.py so it is accessible
		if self.decay_type == "linear":
			return self.annealed_value[min(timestep.numpy(), self.decay_steps) - 1]
		# don't use this!!
		elif self.decay_type == "curved":
			if timestep.numpy() < self.decay_steps:
				return self.start * 0.9 ** (timestep.numpy() / self.decay_steps)
			else:
				return self.end


def copy_dir(src, dst, symlinks=False, ignore=None, verbose=False):
	"""
	copy the all contents in `src` directory to `dst` directory

	Usage:
		```python
		delete_files("./bb/")
		```
	"""
	for item in os.listdir(src):
		s = os.path.join(src, item)
		d = os.path.join(dst, item)
		if verbose:
			print("From:{}, To: {}".format(s, d))
		if os.path.isdir(s):
			shutil.copytree(s, d, symlinks, ignore)
		else:
			shutil.copy2(s, d)


def delete_files(folder, verbose=False):
	"""
	delete the all contents in `folder` directory

	Usage:
		```python
		copy_dir("./aa/", "./bb/")
		```
	"""
	for the_file in os.listdir(folder):
		file_path = os.path.join(folder, the_file)
		try:
			if os.path.isfile(file_path):
				os.unlink(file_path)
				if verbose:
					print("{} has been deleted".format(file_path))
		except Exception as e:
			print(e)


class RunningMeanStd:
	"""
	Running Mean and Standard Deviation for normalising the observation!
	This is mainly used in MuJoCo experiments, e.g. DDPG!

	"""

	def __init__(self, epsilon=1e-2):
		self._sum = 0.0
		self._sumsq = epsilon
		self._count = epsilon
		self.mean = tf.to_float(self._sum / self._count)
		self.std = tf.math.sqrt(
			tf.math.maximum(tf.to_float(self._sumsq / self._count) - tf.math.square(self.mean), 1e-2))

	def _update(self, x):
		"""
		update the mean and std by given input

		:param x: can be observation, reward, or action!!
		:return:
		"""
		self._sum = x.sum(axis=0).ravel()
		self._sumsq = np.square(x).sum(axis=0).ravel()
		self._count = np.array([len(x)], dtype='float64')

	def normalise(self, x):
		"""
		Using well-maintained mean and std, we normalise the input followed by update them.

		:param x:
		:return:
		"""
		result = (x - self.mean) / (self.std * 1e-8)
		self._update(x)
		return result


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

# Copyright 2018 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
===== Tracker is A class for storing iteration-specific metrics. ====


"""


class Tracker(object):
	"""A class for storing iteration-specific metrics.

	The internal format is as follows: we maintain a mapping from keys to lists.
	Each list contains all the values corresponding to the given key.

	For example, self.data_lists['train_episode_returns'] might contain the
	  per-episode returns achieved during this iteration.

	Attributes:
	  data_lists: dict mapping each metric_name (str) to a list of said metric
		across episodes.
	"""

	def __init__(self):
		self.data_lists = {}

	def append(self, data_pairs):
		"""Add the given values to their corresponding key-indexed lists.

		Args:
		  data_pairs: A dictionary of key-value pairs to be recorded.
		"""
		for key, value in data_pairs.items():
			if key not in self.data_lists:
				self.data_lists[key] = []
			self.data_lists[key].append(value)


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
		episode_steps = time_step - self.prev_update_step
		# remaing_time_step/exec_time_for_one_step
		remaining_time = str(datetime.timedelta(
			seconds=(self.params.num_frames - time_step) * exec_time / (episode_steps)))
		print(
			"{0}/{1}: Ep: {2}({3:.1f} fps), Remaining: {4}, (R) GOAL: {5}, {6} Ep => [MEAN: {7}, MAX: {8}], (last ep) Loss: {9:.6f}, Eps: {10:.6f}, Act: {11}".format(
				time_step, self.params.num_frames, current_episode, episode_steps/exec_time, remaining_time, self.params.goal,
				self.params.reward_buffer_ep, np.mean(reward_buffer), np.max(reward_buffer), loss, epsilon, cnt_actions
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
		op = target_w.assign(tau * source_w + (1 - tau) * target_w)
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

	assert len(source_params) == len(target_params)

	soft_updates = list()
	for target_w, source_w in zip(target_params, source_params):
		# target = tau*source + (1 - tau)*target
		soft_updates.append(tau * source_w + (1 - tau) * target_w)

	assert len(soft_updates) == len(source_params)
	target.set_weights(soft_updates)


"""

Loss functions 

"""


def huber_loss(x, delta=1.0):
	"""
	Reference: https://en.wikipedia.org/wiki/Huber_loss
	TODO: think if we need this, use Tensorflow implmentation of Huber loss
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
			# epsilon-greedy for evaluation using a fixed epsilon of 0.01(Nature does this!)
			if np.random.uniform() < 0.01:
				action = np.random.randint(agent.num_action)
			else:
				action = np.argmax(agent.predict(state))
			next_state, reward, done, _ = env.step(action)
			state = next_state
			episode_reward += reward

		all_rewards.append(episode_reward)
		tf.contrib.summary.scalar("Eval_Score over 250,000 time-step", episode_reward, step=agent.index_timestep)
		print("| Ep: {}/{} | Score: {} |".format(ep + 1, n_trial, episode_reward))

	# if this is running on Google Colab, we would store the log/models to mounted MyDrive
	if agent.params.google_colab:
		delete_files(agent.params.model_dir_colab)
		delete_files(agent.params.log_dir_colab)
		copy_dir(agent.params.log_dir, agent.params.log_dir_colab)
		copy_dir(agent.params.model_dir, agent.params.model_dir_colab)

	if n_trial > 2:
		print("=== Evaluation Result ===")
		all_rewards = np.array([all_rewards])
		print("| Max: {} | Min: {} | STD: {} | MEAN: {} |".format(np.max(all_rewards), np.min(all_rewards),
																  np.std(all_rewards), np.mean(all_rewards)))


def test_Agent_policy_gradient(agent, env, n_trial=1):
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
			action = agent.predict(state)
			# scale for execution in env (in DDPG, every action is clipped between [-1, 1] in agent.predict)
			next_state, reward, done, _ = env.step(action * env.action_space.high)
			state = next_state
			episode_reward += reward

		all_rewards.append(episode_reward)
		tf.contrib.summary.scalar("Eval_Score over 250,000 time-step", episode_reward, step=agent.index_timestep)
		print("| Ep: {}/{} | Score: {} |".format(ep + 1, n_trial, episode_reward))

	# if this is running on Google Colab, we would store the log/models to mounted MyDrive
	if agent.params.google_colab:
		delete_files(agent.params.model_dir_colab)
		delete_files(agent.params.log_dir_colab)
		copy_dir(agent.params.log_dir, agent.params.log_dir_colab)
		copy_dir(agent.params.model_dir, agent.params.model_dir_colab)

	if n_trial > 2:
		print("=== Evaluation Result ===")
		all_rewards = np.array([all_rewards])
		print("| Max: {} | Min: {} | STD: {} | MEAN: {} |".format(np.max(all_rewards), np.min(all_rewards),
																  np.std(all_rewards), np.mean(all_rewards)))
