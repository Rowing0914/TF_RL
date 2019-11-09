import argparse
import itertools
import time
from collections import deque

import numpy as np
import tensorflow as tf

from tf_rl.experiments.Exploration_Strategies.utils import eval_Agent, make_grid_env
from tf_rl.experiments.Exploration_Strategies.eval_SAC import Critic
from tf_rl.experiments.Exploration_Strategies.eval_DDPG import Actor

from copy import deepcopy
from tf_rl.common.utils import create_checkpoint
from tf_rl.agents.core import Agent
from tf_rl.common.memory import ReplayBuffer
from tf_rl.common.random_process import OrnsteinUhlenbeckProcess
from tf_rl.common.utils import eager_setup, get_ready, soft_target_model_update_eager, logger

eager_setup()


class DDPG(Agent):
    def __init__(self, actor, critic, num_action, random_process, params):
        self.params = params
        self.num_action = num_action
        self.eval_flg = False
        self.index_timestep = 0
        self.actor = actor(num_action)
        self.critic = critic(1)
        self.target_actor = deepcopy(self.actor)
        self.target_critic = deepcopy(self.critic)
        self.actor_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4)
        self.critic_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
        self.random_process = random_process
        self.actor_manager = create_checkpoint(model=self.actor,
                                               optimizer=self.actor_optimizer,
                                               model_dir=params.actor_model_dir)
        self.critic_manager = create_checkpoint(model=self.critic,
                                                optimizer=self.critic_optimizer,
                                                model_dir=params.critic_model_dir)

    def predict(self, state):
        state = np.expand_dims(state, axis=0).astype(np.float32)
        action = self._select_action(tf.constant(state))
        return action.numpy()[0] + self.random_process.sample()

    def eval_predict(self, state):
        """ Deterministic behaviour """
        state = np.expand_dims(state, axis=0).astype(np.float32)
        action = self._select_action(tf.constant(state))
        return action.numpy()[0]

    @tf.contrib.eager.defun(autograph=False)
    def _select_action(self, state):
        return self.actor(state)

    @tf.contrib.eager.defun(autograph=False)
    def _inner_update(self, states, actions, rewards, next_states, dones):
        self.index_timestep = tf.train.get_global_step()
        # Update Critic
        with tf.GradientTape() as tape:
            # critic takes as input states, actions so that we combine them before passing them
            next_Q1, next_Q2 = self.target_critic(next_states, self.target_actor(next_states))
            min_next_Q_target = tf.math.minimum(next_Q1, next_Q2)
            q1, q2 = self.critic(states, actions)

            # compute the target discounted Q(s', a')
            Y = rewards + self.params.gamma * tf.reshape(min_next_Q_target, [-1]) * (1. - dones)
            Y = tf.stop_gradient(Y)

            # Compute critic loss
            critic_loss_q1 = tf.losses.mean_squared_error(Y, tf.reshape(q1, [-1]))
            critic_loss_q2 = tf.losses.mean_squared_error(Y, tf.reshape(q2, [-1]))

        critic_grads = tape.gradient([critic_loss_q1, critic_loss_q2], self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        # Update Actor
        with tf.GradientTape() as tape:
            actor_loss = -tf.math.reduce_mean(self.critic(states, self.actor(states)))

        # get gradients
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)

        # apply processed gradients to the network
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        return tf.math.reduce_sum(critic_loss_q1 + critic_loss_q2 + actor_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="Cont-GridWorld-v2", type=str, help="Env title")
    parser.add_argument("--train_flg", default="original", type=str, help="train flg: original or on-policy")
    parser.add_argument("--seed", default=123, type=int, help="seed for randomness")
    # parser.add_argument("--num_frames", default=1_000_000, type=int, help="total frame in a training")
    parser.add_argument("--num_frames", default=500_000, type=int, help="total frame in a training")
    parser.add_argument("--eval_interval", default=100_000, type=int, help="a frequency of evaluation in training phase")
    parser.add_argument("--memory_size", default=100_000, type=int, help="memory size in a training")
    parser.add_argument("--learning_start", default=10_000, type=int, help="length before training")
    parser.add_argument("--batch_size", default=200, type=int, help="batch size of each iteration of update")
    parser.add_argument("--reward_buffer_ep", default=10, type=int, help="reward_buffer size")
    parser.add_argument("--gamma", default=0.99, type=float, help="discount factor")
    parser.add_argument("--soft_update_tau", default=1e-2, type=float, help="soft-update tau")
    parser.add_argument("--L2_reg", default=0.5, type=float, help="magnitude of L2 regularisation")
    parser.add_argument("--mu", default=0.3, type=float, help="magnitude of randomness")
    parser.add_argument("--sigma", default=0.2, type=float, help="magnitude of randomness")
    parser.add_argument("--action_range", default=[-1., 1.], type=list, help="magnitude of L2 regularisation")
    parser.add_argument("--debug_flg", default=False, type=bool, help="debug mode or not")
    parser.add_argument("--google_colab", default=False, type=bool, help="if you are executing this on GoogleColab")
    params = parser.parse_args()
    params.test_episodes = 1

    params.log_dir = "./logs/DDPG-two-headed/logs/DDPG-seed{}/".format(params.seed)
    params.actor_model_dir = "./logs/DDPG-two-headed/models/DDPG-seed{}/actor/".format(params.seed)
    params.critic_model_dir = "./logs/DDPG-two-headed/models/DDPG-seed{}/critic/".format(params.seed)
    params.video_dir = "./logs/DDPG-two-headed/video/DDPG-seed{}/".format(params.seed)
    params.plot_path = "./logs/DDPG-two-headed/plots/DDPG-seed{}/".format(params.seed)

    env = make_grid_env(plot_path=params.plot_path)

    # set seed
    env.seed(params.seed)
    tf.random.set_random_seed(params.seed)

    replay_buffer = ReplayBuffer(params.memory_size)
    reward_buffer = deque(maxlen=params.reward_buffer_ep)
    summary_writer = tf.contrib.summary.create_file_writer(params.log_dir)
    random_process = OrnsteinUhlenbeckProcess(size=env.action_space.shape[0], theta=0.15, mu=params.mu, sigma=params.sigma)
    agent = DDPG(Actor, Critic, env.action_space.shape[0], random_process, params)

    get_ready(agent.params)

    global_timestep = tf.compat.v1.train.get_or_create_global_step()
    time_buffer = deque(maxlen=agent.params.reward_buffer_ep)
    log = logger(agent.params)

    traj = list()

    with summary_writer.as_default():
        # for summary purpose, we put all codes in this context
        with tf.contrib.summary.always_record_summaries():

            for i in itertools.count():
                state = env.reset()
                total_reward = 0
                self_rewards = 0
                start = time.time()
                agent.random_process.reset_states()
                done = False
                episode_len = 0
                while not done:
                    traj.append(state)
                    if global_timestep.numpy() < agent.params.learning_start:
                        action = env.action_space.sample()
                    else:
                        action = agent.predict(state)
                    next_state, reward, done, info = env.step(np.clip(action, -1.0, 1.0))
                    replay_buffer.add(state, action, reward, next_state, done)

                    """
                    === Update the models
                    """
                    if global_timestep.numpy() > agent.params.learning_start:
                        states, actions, rewards, next_states, dones = replay_buffer.sample(agent.params.batch_size)
                        loss = agent.update(states, actions, rewards, next_states, dones)
                        soft_target_model_update_eager(agent.target_actor, agent.actor, tau=agent.params.soft_update_tau)
                        soft_target_model_update_eager(agent.target_critic, agent.critic, tau=agent.params.soft_update_tau)

                    global_timestep.assign_add(1)
                    episode_len += 1
                    total_reward += reward
                    state = next_state

                    # for evaluation purpose
                    if global_timestep.numpy() % agent.params.eval_interval == 0:
                        agent.eval_flg = True

                """
                ===== After 1 Episode is Done =====
                """
                # save the updated models
                agent.actor_manager.save()
                agent.critic_manager.save()

                # store the episode related variables
                reward_buffer.append(total_reward)
                time_buffer.append(time.time() - start)

                # logging on Tensorboard
                tf.contrib.summary.scalar("reward", total_reward, step=global_timestep.numpy())
                tf.contrib.summary.scalar("exec time", time.time() - start, step=global_timestep.numpy())
                if i >= agent.params.reward_buffer_ep:
                    tf.contrib.summary.scalar("Moving Ave Reward", np.mean(reward_buffer), step=global_timestep.numpy())

                # logging
                if global_timestep.numpy() > agent.params.learning_start and i % agent.params.reward_buffer_ep == 0:
                    log.logging(global_timestep.numpy(), i, np.sum(time_buffer), reward_buffer, np.mean(loss), 0, [0])

                # evaluation
                if agent.eval_flg:
                    env.vis_exploration(traj=np.array(traj),
                                        file_name="exploration_train_DDPG_{}.png".format(global_timestep.numpy()))
                    env.vis_trajectory(traj=np.array(traj), file_name="traj_train_DDPG_{}.png".format(global_timestep.numpy()))
                    eval_Agent(env, agent)
                    agent.eval_flg = False

                # check the stopping condition
                if global_timestep.numpy() > agent.params.num_frames:
                    print("=== Training is Done ===")
                    traj = np.array(traj)
                    env.vis_exploration(traj=traj, file_name="exploration_DDPG_during_training.png")
                    env.vis_trajectory(traj=traj, file_name="traj_DDPG_during_training.png")
                    eval_Agent(env, agent)
                    env.close()
                    break
