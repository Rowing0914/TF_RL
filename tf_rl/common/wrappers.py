import numpy as np
import os, math

os.environ.setdefault('PATH', '')
from collections import deque
import gym
from gym import spaces

# for those who installed ROS on local env
import sys

try:
	sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
	pass

import cv2

cv2.ocl.setUseOpenCL(False)

"""
Wrapper for Cartpole
This is to change the reward at the terminal state because originally it is set as 1.0

check here: https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py

"""


class MyWrapper(gym.Wrapper):
	def __init__(self, env):
		gym.Wrapper.__init__(self, env)
		self.env = env
		self.env.seed(123)  # fix the randomness for reproducibility purpose

	def step(self, ac):
		observation, reward, done, info = self.env.step(ac)
		if done:
			reward = -1.0  # reward at a terminal state
		return observation, reward, done, info

	def reset(self, **kwargs):
		return self.env.reset()


class DiscretisedEnv(gym.Wrapper):
	def __init__(self, env):
		gym.Wrapper.__init__(self, env)
		self.env = env
		self.high1 = env.observation_space.high[0]
		self.high2 = env.observation_space.high[2]
		self.low1 = env.observation_space.low[0]
		self.low2 = env.observation_space.low[2]
		self.buckets = (1, 1, 6, 12,)
		self.env.seed(123)  # fix the randomness for reproducibility purpose

	def step(self, ac):
		observation, reward, done, info = self.env.step(ac)
		if done:
			reward = -1.0  # reward at a terminal state
		return self._discretise(observation), reward, done, info

	def reset(self, **kwargs):
		return self._discretise(self.env.reset())

	def _discretise(self, obs):
		upper_bounds = [self.high1, 0.5, self.high2, math.radians(50)]
		lower_bounds = [self.low1, -0.5, self.low2, -math.radians(50)]
		ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in
				  range(len(obs))]
		new_obs = [int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
		new_obs = [min(self.buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
		return tuple(new_obs)


class MyWrapper_revertable(gym.Wrapper):
	def __init__(self, env):
		gym.Wrapper.__init__(self, env)
		self.env = env.unwrapped
		self.env.seed(123)  # fix the randomness for reproducibility purpose

	def step(self, ac):
		next_state, reward, done, info = self.env.step(ac)
		if done:
			reward = -1.0  # reward at a terminal state
		return next_state, reward, done, info

	def reset(self, **kwargs):
		return self.env.reset()

	def get_state(self):
		return self.env.state

	def set_state(self, state):
		self.env.state = state


"""
Borrowed from OpenAI Baselines at 4/4/2019

https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
https://github.com/openai/baselines/blob/master/baselines/common/wrappers.py

"""


# I moved the source code of this module just below
# from .wrappers import TimeLimit

class TimeLimit(gym.Wrapper):
	def __init__(self, env, max_episode_steps=None):
		super(TimeLimit, self).__init__(env)
		self._max_episode_steps = max_episode_steps
		self._elapsed_steps = 0

	def step(self, ac):
		observation, reward, done, info = self.env.step(ac)
		self._elapsed_steps += 1
		if self._elapsed_steps >= self._max_episode_steps:
			done = True
			info['TimeLimit.truncated'] = True
		return observation, reward, done, info

	def reset(self, **kwargs):
		self._elapsed_steps = 0
		return self.env.reset(**kwargs)


class NoopResetEnv(gym.Wrapper):
	def __init__(self, env, noop_max=30):
		"""Sample initial states by taking random number of no-ops on reset.
		No-op is assumed to be action 0.
		"""
		gym.Wrapper.__init__(self, env)
		self.noop_max = noop_max
		self.override_num_noops = None
		self.noop_action = 0
		assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

	def reset(self, **kwargs):
		""" Do no-op action for a number of steps in [1, noop_max]."""
		self.env.reset(**kwargs)
		if self.override_num_noops is not None:
			noops = self.override_num_noops
		else:
			noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)  # pylint: disable=E1101
		assert noops > 0
		obs = None
		for _ in range(noops):
			obs, _, done, _ = self.env.step(self.noop_action)
			if done:
				obs = self.env.reset(**kwargs)
		return obs

	def step(self, ac):
		return self.env.step(ac)


class FireResetEnv(gym.Wrapper):
	def __init__(self, env):
		"""Take action on reset for environments that are fixed until firing."""
		gym.Wrapper.__init__(self, env)
		assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
		assert len(env.unwrapped.get_action_meanings()) >= 3

	def reset(self, **kwargs):
		self.env.reset(**kwargs)
		obs, _, done, _ = self.env.step(1)
		if done:
			self.env.reset(**kwargs)
		obs, _, done, _ = self.env.step(2)
		if done:
			self.env.reset(**kwargs)
		return obs

	def step(self, ac):
		return self.env.step(ac)


class EpisodicLifeEnv(gym.Wrapper):
	def __init__(self, env):
		"""Make end-of-life == end-of-episode, but only reset on true game over.
		Done by DeepMind for the DQN and co. since it helps value estimation.
		"""
		gym.Wrapper.__init__(self, env)
		self.lives = 0
		self.was_real_done = True

	def step(self, action):
		obs, reward, done, info = self.env.step(action)
		self.was_real_done = done
		# check current lives, make loss of life terminal,
		# then update lives to handle bonus lives
		lives = self.env.unwrapped.ale.lives()
		if lives < self.lives and lives > 0:
			# for Qbert sometimes we stay in lives == 0 condition for a few frames
			# so it's important to keep lives > 0, so that we only reset once
			# the environment advertises done.
			done = True
		self.lives = lives
		return obs, reward, done, info

	def reset(self, **kwargs):
		"""Reset only when lives are exhausted.
		This way all states are still reachable even though lives are episodic,
		and the learner need not know about any of this behind-the-scenes.
		"""
		if self.was_real_done:
			obs = self.env.reset(**kwargs)
		else:
			# no-op step to advance from terminal/lost life state
			obs, _, _, _ = self.env.step(0)
		self.lives = self.env.unwrapped.ale.lives()
		return obs


class MaxAndSkipEnv(gym.Wrapper):
	def __init__(self, env, skip=4):
		"""Return only every `skip`-th frame"""
		gym.Wrapper.__init__(self, env)
		# most recent raw observations (for max pooling across time steps)
		self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
		self._skip = skip

	def step(self, action):
		"""Repeat action, sum reward, and max over last observations."""
		total_reward = 0.0
		done = None
		for i in range(self._skip):
			obs, reward, done, info = self.env.step(action)
			if i == self._skip - 2: self._obs_buffer[0] = obs
			if i == self._skip - 1: self._obs_buffer[1] = obs
			total_reward += reward
			if done:
				break
		# Note that the observation on the done=True frame
		# doesn't matter
		max_frame = self._obs_buffer.max(axis=0)

		return max_frame, total_reward, done, info

	def reset(self, **kwargs):
		return self.env.reset(**kwargs)


class ClipRewardEnv(gym.RewardWrapper):
	def __init__(self, env):
		gym.RewardWrapper.__init__(self, env)

	def reward(self, reward):
		"""
		Bin reward to {+1, 0, -1} by its sign.

		Reference:
			https://docs.scipy.org/doc/numpy-1.9.3/reference/generated/numpy.sign.html

		Usage:
			>>> np.sign([-5., 4.5])
			array([-1.,  1.])
			>>> np.sign(0)
			0

		"""
		return np.sign(reward)


class WarpFrame(gym.ObservationWrapper):
	def __init__(self, env, width=84, height=84, grayscale=True):
		"""Warp frames to 84x84 as done in the Nature paper and later work."""
		gym.ObservationWrapper.__init__(self, env)
		self.width = width
		self.height = height
		self.grayscale = grayscale
		if self.grayscale:
			self.observation_space = spaces.Box(low=0, high=255,
												shape=(self.height, self.width, 1), dtype=np.uint8)
		else:
			self.observation_space = spaces.Box(low=0, high=255,
												shape=(self.height, self.width, 3), dtype=np.uint8)

	def observation(self, frame):
		if self.grayscale:
			frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
		frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
		if self.grayscale:
			frame = np.expand_dims(frame, -1)
		return frame


class FrameStack(gym.Wrapper):
	def __init__(self, env, k):
		"""Stack k last frames.

		Returns lazy array, which is much more memory efficient.

		See Also
		--------
		baselines.common.atari_wrappers.LazyFrames
		"""
		gym.Wrapper.__init__(self, env)
		self.k = k
		self.frames = deque([], maxlen=k)
		shp = env.observation_space.shape
		self.observation_space = spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)),
											dtype=env.observation_space.dtype)

	def reset(self):
		ob = self.env.reset()
		for _ in range(self.k):
			self.frames.append(ob)
		return self._get_ob()

	def step(self, action):
		ob, reward, done, info = self.env.step(action)
		self.frames.append(ob)
		return self._get_ob(), reward, done, info

	def _get_ob(self):
		assert len(self.frames) == self.k
		# don't do this.... LazyFrame is much more memory efficient
		# on my local, it reduced from 1.4GB for ReplayBuffer(50000) to 397.1MB..... incredible.
		# return np.concatenate(list(self.frames), axis=-1)
		return LazyFrames(list(self.frames))


class ScaledFloatFrame(gym.ObservationWrapper):
	def __init__(self, env):
		gym.ObservationWrapper.__init__(self, env)
		self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

	def observation(self, observation):
		# careful! This undoes the memory optimization, use
		# with smaller replay buffers only.
		return np.array(observation).astype(np.float32) / 255.0


class LazyFrames(object):
	def __init__(self, frames):
		"""This object ensures that common frames between the observations are only stored once.
		It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
		buffers.

		This object should only be converted to numpy array before being passed to the model.

		You'd not believe how complex the previous solution was."""
		self._frames = frames
		self._out = None

	def _force(self):
		if self._out is None:
			self._out = np.concatenate(self._frames, axis=-1)
			self._frames = None
		return self._out

	def __array__(self, dtype=None):
		out = self._force()
		if dtype is not None:
			out = out.astype(dtype)
		return out

	def __len__(self):
		return len(self._force())

	def __getitem__(self, i):
		return self._force()[i]


def make_atari(env_id, max_episode_steps=None):
	env = gym.make(env_id)
	assert 'NoFrameskip' in env.spec.id
	env = NoopResetEnv(env, noop_max=30)
	env = MaxAndSkipEnv(env, skip=4)
	if max_episode_steps is not None:
		env = TimeLimit(env, max_episode_steps=max_episode_steps)
	return env

# since my code does not have an function or APIs to repeat the same action several times,
# I will rely on those wrappers.
def wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=True, scale=False):
	"""Configure environment for DeepMind-style Atari.
	"""
	if episode_life:
		env = EpisodicLifeEnv(env)
	if 'FIRE' in env.unwrapped.get_action_meanings():
		env = FireResetEnv(env)
	env = WarpFrame(env)
	if scale:
		env = ScaledFloatFrame(env)
	if clip_rewards:
		env = ClipRewardEnv(env)
	if frame_stack:
		env = FrameStack(env, 4)
	return env
