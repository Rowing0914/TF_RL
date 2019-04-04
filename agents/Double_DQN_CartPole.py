import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from common.memory import ReplayBuffer
from common.utils import AnnealingEpsilon, sync_main_target
from DQN import DQN_CartPole, Parameters


env = gym.make("CartPole-v0")
params = Parameters(mode="CartPole")

# log purpose
losses = []
all_rewards = []
episode_reward = 0

# initialise a graph in a session
tf.reset_default_graph()

# training epoch
global_step = tf.Variable(0, name="global_step", trainable=False)

# initialise models and replay memory
main_model = DQN_CartPole("main", env)
target_model = DQN_CartPole("target", env)
replay_buffer = ReplayBuffer(params.memory_size)
Epsilon = AnnealingEpsilon(start=params.epsilon_start, end=params.epsilon_end, decay_steps=params.decay_steps)

with tf.Session() as sess:
	# initialise all variables used in the model
	sess.run(tf.global_variables_initializer())
	state = env.reset()
	for frame_idx in range(1, params.num_frames + 1):
		action = target_model.act(sess, state.reshape(1, 4), Epsilon.get_epsilon(frame_idx))

		next_state, reward, done, _ = env.step(action)
		replay_buffer.store(state, action, reward, next_state, done)

		state = next_state
		episode_reward += reward

		if done:
			state = env.reset()
			all_rewards.append(episode_reward)
			print("\rGAME OVER AT STEP: {0}, SCORE: {1}".format(frame_idx, episode_reward), end="")
			episode_reward = 0

			if frame_idx > params.learning_start:
				if len(replay_buffer) > params.batch_size:
					states, actions, rewards, next_states, dones = replay_buffer.sample(params.batch_size)
					next_Q_main = main_model.predict(sess, next_states)
					next_Q = target_model.predict(sess, next_states)
					Y = rewards + params.gamma * next_Q[np.arange(params.batch_size), np.argmax(next_Q_main, axis=1)] * dones
					# print(Y)
					loss = main_model.update(sess, states, actions, Y)
					losses.append(loss)
			else:
				pass

		if frame_idx > params.learning_start:
			if frame_idx % params.sync_freq == 0:
				print("\nModel Sync")
				sync_main_target(sess, main_model, target_model)

# temporal visualisation
plt.subplot(2, 1, 1)
plt.plot(all_rewards)
plt.title("Score over time")
plt.xlabel("Timestep")
plt.ylabel("Score")

plt.subplot(2, 1, 2)
plt.plot(losses)
plt.title("Loss over time")
plt.xlabel("Timestep")
plt.ylabel("Loss")
plt.show()