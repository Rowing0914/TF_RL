import numpy as np
import tensorflow as tf
from tf_rl.common.utils import create_checkpoint

class Agent_atari:
	"""
	boiler plate of an agent for Atari game

	"""

	def __init__(self, model, optimizer, loss_fn, grad_clip_fn, num_action, gamma, model_dir):
		self.num_action = num_action
		self.gamma = gamma
		self.grad_clip_fn = grad_clip_fn
		self.loss_fn = loss_fn
		self.eval_flg = False
		self.index_timestep = 0
		self.main_model = model(num_action)
		self.target_model = model(num_action)
		self.optimizer = optimizer
		self.manager = create_checkpoint(model=self.main_model,
										 optimizer=self.optimizer,
										 model_dir=model_dir)

	def predict(self, state):
		state = np.expand_dims(state / 255., axis=0).astype(np.float32)
		action = self._select_action(tf.constant(state))
		return action.numpy()[0]

	def _select_action(self, state):
		raise NotImplementedError

	def update(self, states, actions, rewards, next_states, dones):
		states = np.array(states, dtype=np.float32)
		next_states = np.array(next_states, dtype=np.float32)
		actions = np.array(actions, dtype=np.uint8)
		rewards = np.array(rewards, dtype=np.float32)
		dones = np.array(dones, dtype=np.float32)
		return self.inner_update(states, actions, rewards, next_states, dones)

	def inner_update(self, states, actions, rewards, next_states, dones):
		raise NotImplementedError


class Agent_cartpole:
	"""
	boiler plate of an agent for Atari game

	"""

	def __init__(self, model, optimizer, loss_fn, grad_clip_fn, num_action, gamma, model_dir):
		self.num_action = num_action
		self.gamma = gamma
		self.grad_clip_fn = grad_clip_fn
		self.loss_fn = loss_fn
		self.eval_flg = False
		self.index_timestep = 0
		self.main_model = model(num_action)
		self.target_model = model(num_action)
		self.optimizer = optimizer
		self.manager = create_checkpoint(model=self.main_model,
										 optimizer=self.optimizer,
										 model_dir=model_dir)

	def predict(self, state):
		state = np.expand_dims(state, axis=0).astype(np.float32)
		action = self._select_action(tf.constant(state))
		return action.numpy()[0]

	def _select_action(self, state):
		raise NotImplementedError

	def update(self, states, actions, rewards, next_states, dones):
		states = np.array(states, dtype=np.float32)
		next_states = np.array(next_states, dtype=np.float32)
		actions = np.array(actions, dtype=np.uint8)
		rewards = np.array(rewards, dtype=np.float32)
		dones = np.array(dones, dtype=np.float32)
		return self.inner_update(states, actions, rewards, next_states, dones)

	def inner_update(self, states, actions, rewards, next_states, dones):
		raise NotImplementedError