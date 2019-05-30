import time, sys, datetime
import tensorflow as tf
from tf_rl.common.utils import logger


class Runner:
	def __init__(self, agent, env, args):
		self.agent = agent
		self.env = env
		self.args = args
		self.global_timestep = tf.train.get_or_create_global_step()
		self.args.max_episode_steps = args.max_episode_steps if args.max_episode_steps else 27000  # borrow from dopamine
		self.summary_writer = tf.contrib.summary.create_file_writer(args.log_dir)
		# TODO: maybe we don't need to define session.
		self.sess = self._get_session(eager=True)
		self.log = logger(args)

	def _get_session(self, eager=True):
		if eager:

			return None
		else:
			config = tf.ConfigProto(allow_soft_placement=True)
			config.gpu_options.allow_growth = True
			sess = tf.Session(config=config)
			sess.run(tf.global_variables_initializer())
			return sess

	def _initialize_episode(self):
		"""Initialization for a new episode.

		Returns:
		  action: int, the initial action chosen by the agent.
		"""
		initial_observation = self.env.reset()
		return self.agent.begin_episode(initial_observation)

	def _end_episode(self, reward):
		"""Finalizes an episode run.

		Args:
		  reward: float, the last reward from the environment.
		"""
		self.agent.end_episode(reward)

	def _run_one_episode(self):
		"""Executes a full trajectory of the agent interacting with the environment.

		Returns:
		  The number of steps taken and the total reward.
		"""
		step_number = 0
		total_reward = 0.

		action = self._initialize_episode()
		is_terminal = False

		# Keep interacting until we reach a terminal state.
		while True:
			observation, reward, is_terminal, _ = self.env.step(action)

			tf.assign(self.global_timestep, self.global_timestep.numpy() + 1, name='update_global_step')
			total_reward += reward
			step_number += 1

			# TODO: check if wrapper handles clipping
			# reward = np.clip(reward, -1, 1)

			if (self.env.game_over or
					step_number == self.args.max_episode_steps):
				# Stop the run loop once we reach the true end of episode.
				break
			elif is_terminal:
				# If we lose a life but the episode is not over, signal an artificial
				# end of episode to the agent.
				self.agent.end_episode(reward)
				action = self.agent.begin_episode(observation)
			else:
				action = self.agent.step(reward, observation)

		self._end_episode(reward)

		return total_reward

	def _run_one_phase(self, min_steps):
		"""Runs the agent/environment loop until a desired number of steps.

		We follow the Machado et al., 2017 convention of running full episodes,
		and terminating once we've run a minimum number of steps.

		Args:
		  min_steps: int, minimum number of steps to generate in this phase <= train or eval
		  statistics: `IterationStatistics` object which records the experimental
			results.
		  run_mode_str: str, describes the run mode for this agent.

		Returns:
		  Tuple containing the number of steps taken in this phase (int), the sum of
			returns (float), and the number of episodes performed (int).
		"""
		step_count = 0
		num_episodes = 0
		sum_returns = 0.

		while step_count < min_steps:
			episode_return = self._run_one_episode()
			sum_returns += episode_return
			num_episodes += 1
			# We use sys.stdout.write instead of tf.logging so as to flush frequently
			# without generating a line break.
			sys.stdout.write('Steps executed: {} '.format(self.global_timestep.numpy()) +
							 'Episode number: {} '.format(num_episodes) +
							 'Return: {}\r'.format(episode_return))
			sys.stdout.flush()
		return step_count, sum_returns, num_episodes

	def _run_train_phase(self):
		"""Run training phase.

		Args:
		  statistics: `IterationStatistics` object which records the experimental
			results. Note - This object is modified by this method.

		Returns:
		  num_episodes: int, The number of episodes run in this phase.
		  average_reward: The average reward generated in this phase.
		"""
		# Perform the training phase, during which the agent learns.
		self.agent.eval_mode = False
		start_time = time.time()
		number_steps, sum_returns, num_episodes = self._run_one_phase(self.args.train_steps)
		average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0
		time_delta = time.time() - start_time
		remaining_time = str(datetime.timedelta(
			seconds=(self.args.num_frames - self.global_timestep.numpy()) * time_delta / number_steps))
		tf.logging.info('Average undiscounted return per training episode: %.2f',
						average_return)
		tf.logging.info('Average training steps per second: %.2f',
						number_steps / time_delta)
		tf.logging.info('Remaining Time: {}'.format(remaining_time))
		return num_episodes, average_return

	def _run_eval_phase(self):
		"""Run evaluation phase.

		Args:
		  statistics: `IterationStatistics` object which records the experimental
			results. Note - This object is modified by this method.

		Returns:
		  num_episodes: int, The number of episodes run in this phase.
		  average_reward: float, The average reward generated in this phase.
		"""
		# Perform the evaluation phase -- no learning.
		self.agent.eval_mode = True
		_, sum_returns, num_episodes = self._run_one_phase(self.args.eval_steps)
		average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0
		tf.logging.info('Average undiscounted return per evaluation episode: %.2f', average_return)
		return num_episodes, average_return

	def _save_tensorboard_summaries(self, iteration,
									num_episodes_train,
									average_reward_train,
									num_episodes_eval,
									average_reward_eval):
		"""Save statistics as tensorboard summaries.

		Args:
		  iteration: int, The current iteration number.
		  num_episodes_train: int, number of training episodes run.
		  average_reward_train: float, The average training reward.
		  num_episodes_eval: int, number of evaluation episodes run.
		  average_reward_eval: float, The average evaluation reward.
		"""
		summary = tf.Summary(value=[
			tf.Summary.Value(tag='Train/NumEpisodes',
							 simple_value=num_episodes_train),
			tf.Summary.Value(tag='Train/AverageReturns',
							 simple_value=average_reward_train),
			tf.Summary.Value(tag='Eval/NumEpisodes',
							 simple_value=num_episodes_eval),
			tf.Summary.Value(tag='Eval/AverageReturns',
							 simple_value=average_reward_eval)
		])
		self.summary_writer.add_summary(summary, iteration)

	def _checkpoint_experiment(self):
		"""Checkpoint experiment data.

		Args:
		  iteration: int, iteration number for checkpointing.
		"""
		# TF: checkpoint vs Saver => https://stackoverflow.com/questions/53569622/difference-between-tf-train-checkpoint-and-tf-train-saver
		self.checkpoint_dir = self.args.model_dir
		self.check_point = tf.train.Checkpoint(optimizer=self.agent.optimizer,
											   model=self.agent.main_model,
											   optimizer_step=tf.train.get_or_create_global_step())
		self.manager = tf.train.CheckpointManager(self.check_point, self.checkpoint_dir, max_to_keep=3)

		# try re-loading the previous training progress!
		try:
			print("Try loading the previous training progress")
			self.check_point.restore(self.manager.latest_checkpoint)
			assert tf.train.get_global_step().numpy() != 0
			print("===================================================\n")
			print("Restored the model from {}".format(self.checkpoint_dir))
			print("Currently we are on time-step: {}".format(tf.train.get_global_step().numpy()))
			print("\n===================================================")
		except:
			print("===================================================\n")
			print("Previous Training files are not found in Directory: {}".format(self.checkpoint_dir))
			print("\n===================================================")

	def run_experiment(self):
		"""Runs a full experiment, spread over multiple iterations."""
		tf.logging.info('Beginning training...')
		iteration = 0
		while self.args.num_frames >= self.global_timestep.numpy():
			tf.logging.info('Starting iteration %d', iteration)
			num_episodes_train, average_reward_train = self._run_train_phase()
			num_episodes_eval, average_reward_eval = self._run_eval_phase()

			self._save_tensorboard_summaries(iteration, num_episodes_train,
											 average_reward_train, num_episodes_eval,
											 average_reward_eval)
			iteration += 1
