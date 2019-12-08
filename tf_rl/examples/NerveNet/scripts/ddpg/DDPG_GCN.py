import gym
import argparse
import time
import tensorflow as tf
from collections import deque
from tf_rl.common.monitor import Monitor
from tf_rl.common.random_process import OrnsteinUhlenbeckProcess, GaussianNoise
from tf_rl.common.memory import ReplayBuffer
from tf_rl.common.utils import *
from tf_rl.common.networks import DDPG_Critic as Critic
from tf_rl.common.visualise import visualise_act_and_dist

from graph_util.mujoco_parser import get_adjacency_matrix, get_adjacency_matrix_Ant
from network.gcn import GCN
import environments.register as register

eager_setup()

from copy import deepcopy

class DDPG:
    def __init__(self, gcn, critic, adjacency_matrix, num_node_feature, num_action, random_process, params):
        self.params = params
        self.num_action = num_action
        self.eval_flg = False
        self.index_timestep = 0
        self.actor = gcn(adjacency_matrix, num_node_feature)
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
        action = self._select_action(tf.constant(state.astype(np.float32)))
        return action.numpy().flatten() + self.random_process.sample()

    def eval_predict(self, state):
        """ Deterministic behaviour """
        action = self._select_action(tf.constant(state.astype(np.float32)))
        return action.numpy().flatten()

    @tf.contrib.eager.defun(autograph=False)
    def _select_action(self, state):
        return self.actor(state)

    def update(self, states, actions, rewards, next_states, dones):
        """
        Update methods for Actor and Critic
        please refer to https://arxiv.org/pdf/1509.02971.pdf about the details

        """
        states = np.array(states, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        actions = np.array(actions, dtype=np.float32)[..., np.newaxis]
        rewards = np.array(rewards, dtype=np.float32)[..., np.newaxis]
        dones = np.array(dones, dtype=np.float32)[..., np.newaxis]
        return self._inner_update(states, actions, rewards, next_states, dones)

    @tf.contrib.eager.defun(autograph=False)
    def _inner_update(self, states, actions, rewards, next_states, dones):
        self.index_timestep = tf.compat.v1.train.get_global_step()
        # Update Critic
        with tf.GradientTape() as tape:
            # critic takes as input states, actions so that we combine them before passing them
            a = self.target_actor(next_states)
            next_Q = self.target_critic(next_states, self.target_actor(next_states))
            q_values = self.critic(states, actions)

            # compute the target discounted Q(s', a')
            Y = rewards + self.params.gamma * tf.compat.v1.squeeze(next_Q, [-1]) * (1. - dones)
            Y = tf.stop_gradient(Y)

            # Compute critic loss(MSE or huber_loss) + L2 loss
            critic_loss = tf.compat.v1.losses.mean_squared_error(Y, tf.compat.v1.squeeze(q_values, [-1])) + tf.add_n(
                self.critic.losses) * self.params.L2_reg

        # get gradients
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)

        # apply processed gradients to the network
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        # Update Actor
        with tf.GradientTape() as tape:
            actor_loss = -tf.math.reduce_mean(self.critic(states, self.actor(states)))

        # get gradients
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)

        # apply processed gradients to the network
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        return np.sum(critic_loss + actor_loss)

def eval_Agent_DDPG(env, agent, n_trial=1):
    """
    Evaluate the trained agent with the recording of its behaviour

    :return:
    """

    all_distances, all_rewards, all_actions = list(), list(), list()
    distance_func = get_distance(agent.params.env_name) # create the distance measure func
    print("=== Evaluation Mode ===")
    for ep in range(n_trial):
        env.record_start()
        obs = env.reset()
        state = obs["graph_obs"]
        done = False
        episode_reward = 0
        while not done:
            action = agent.eval_predict(state)
            # scale for execution in env (in DDPG, every action is clipped between [-1, 1] in agent.predict)
            obs, reward, done, info = env.step(action * env.action_space.high)
            next_flat_state, next_graph_state = obs["flat_obs"], obs["graph_obs"]
            distance = distance_func(action, reward, info)
            all_actions.append(action.mean()**2) # Mean Squared of action values
            all_distances.append(distance)
            state = next_graph_state
            episode_reward += reward

        all_rewards.append(episode_reward)
        tf.contrib.summary.scalar("Evaluation Score", episode_reward, step=agent.index_timestep)
        print("| Ep: {}/{} | Score: {} |".format(ep + 1, n_trial, episode_reward))
    env.record_end()
    return all_rewards, all_distances, all_actions


parser = argparse.ArgumentParser()
parser.add_argument("--env_name", default="AntWithGoal-v1", type=str, help="Env title")
parser.add_argument("--seed", default=123, type=int, help="seed for randomness")
# parser.add_argument("--num_frames", default=1_000_000, type=int, help="total frame in a training")
parser.add_argument("--num_frames", default=500_000, type=int, help="total frame in a training")
# parser.add_argument("--num_frames", default=20_000, type=int, help="total frame in a training")
parser.add_argument("--eval_interval", default=100_000, type=int, help="a frequency of evaluation in training phase")
# parser.add_argument("--eval_interval", default=5_000, type=int, help="a frequency of evaluation in training phase")
parser.add_argument("--memory_size", default=100_000, type=int, help="memory size in a training")
parser.add_argument("--learning_start", default=10_000, type=int, help="length before training")
# parser.add_argument("--learning_start", default=1_000, type=int, help="length before training")
parser.add_argument("--batch_size", default=200, type=int, help="batch size of each iteration of update")
parser.add_argument("--reward_buffer_ep", default=10, type=int, help="reward_buffer size")
parser.add_argument("--gamma", default=0.99, type=float, help="discount factor")
parser.add_argument("--soft_update_tau", default=1e-2, type=float, help="soft-update tau")
parser.add_argument("--L2_reg", default=0.5, type=float, help="magnitude of L2 regularisation")
parser.add_argument("--mu", default=0.2, type=float, help="magnitude of randomness")
parser.add_argument("--sigma", default=0.05, type=float, help="magnitude of randomness")
parser.add_argument("--action_range", default=[-1., 1.], type=list, help="magnitude of L2 regularisation")
parser.add_argument("--debug_flg", default=False, type=bool, help="debug mode or not")
parser.add_argument("--google_colab", default=False, type=bool, help="if you are executing this on GoogleColab")
params = parser.parse_args()
params.test_episodes = 1
params.goal = 0

now = datetime.datetime.now()

params.log_dir = "../logs/logs/" + now.strftime("%Y%m%d-%H%M%S") + "-DDPG-GCN/"
params.actor_model_dir = "../logs/models/" + now.strftime("%Y%m%d-%H%M%S") + "-DDPG-GCN_actor/"
params.critic_model_dir = "../logs/models/" + now.strftime("%Y%m%d-%H%M%S") + "-DDPG-GCN_critic/"
params.video_dir = "../logs/video/GCN_{}".format(now.strftime("%Y%m%d-%H%M%S") + "_" + str(params.env_name))
params.plot_path = "../logs/plots/GCN_{}/".format(now.strftime("%Y%m%d-%H%M%S") + "_" + str(params.env_name))

env = gym.make(params.env_name)
env = Monitor(env, params.video_dir, force=True)

# set seed
env.seed(params.seed)
tf.random.set_random_seed(params.seed)

replay_buffer = ReplayBuffer(params.memory_size)
reward_buffer = deque(maxlen=params.reward_buffer_ep)
summary_writer = tf.contrib.summary.create_file_writer(params.log_dir)
random_process = OrnsteinUhlenbeckProcess(size=env.action_space.shape[0], theta=0.15, mu=params.mu, sigma=params.sigma)
# random_process = GaussianNoise(mu=params.mu, sigma=params.sigma)

# adjacency_matrix = get_adjacency_matrix(num_legs=4)
adjacency_matrix = get_adjacency_matrix_Ant()
num_node_features = 128
agent = DDPG(GCN, Critic, adjacency_matrix, num_node_features, env.action_space.shape[0], random_process, params)

get_ready(agent.params)

global_timestep = tf.compat.v1.train.get_or_create_global_step()
time_buffer = deque(maxlen=agent.params.reward_buffer_ep)
log = logger(agent.params)
action_buffer, distance_buffer, eval_epochs = list(), list(), list()

with summary_writer.as_default():
    # for summary purpose, we put all codes in this context
    with tf.contrib.summary.always_record_summaries():

        for i in itertools.count():
            obs = env.reset()
            state = obs["graph_obs"]
            total_reward = 0
            start = time.time()
            agent.random_process.reset_states()
            done = False
            episode_len = 0
            while not done:
                if global_timestep.numpy() < agent.params.learning_start:
                    action = env.action_space.sample()
                else:
                    action = agent.predict(state)
                # scale for execution in env (in DDPG, every action is clipped between [-1, 1] in agent.predict)
                obs, reward, done, info = env.step(action * env.action_space.high)
                next_flat_state, next_graph_state = obs["flat_obs"], obs["graph_obs"]
                replay_buffer.add(state, action, reward, next_graph_state, done)

                global_timestep.assign_add(1)
                episode_len += 1
                total_reward += reward
                state = next_graph_state

                # for evaluation purpose
                if global_timestep.numpy() % agent.params.eval_interval == 0:
                    agent.eval_flg = True

            """
            ===== After 1 Episode is Done =====
            """

            # train the model at this point
            for t_train in range(episode_len):
                # for t_train in range(10): # for test purpose
                states, actions, rewards, next_states, dones = replay_buffer.sample(agent.params.batch_size)
                loss = agent.update(states, actions, rewards, next_states, dones)
                soft_target_model_update_eager(agent.target_actor, agent.actor, tau=agent.params.soft_update_tau)
                soft_target_model_update_eager(agent.target_critic, agent.critic, tau=agent.params.soft_update_tau)

            # save the update models
            agent.actor_manager.save()
            agent.critic_manager.save()

            # store the episode related variables
            reward_buffer.append(total_reward)
            time_buffer.append(time.time() - start)

            # logging on Tensorboard
            tf.contrib.summary.scalar("reward", total_reward, step=i)
            tf.contrib.summary.scalar("exec time", time.time() - start, step=i)
            if i >= agent.params.reward_buffer_ep:
                tf.contrib.summary.scalar("Moving Ave Reward", np.mean(reward_buffer), step=i)

            # logging
            if global_timestep.numpy() > agent.params.learning_start and i % agent.params.reward_buffer_ep == 0:
                log.logging(global_timestep.numpy(), i, np.sum(time_buffer), reward_buffer, np.mean(loss), 0, [0])

            # evaluation
            if agent.eval_flg:
                eval_reward, eval_distance, eval_action = eval_Agent_DDPG(env, agent)
                eval_epochs.append(global_timestep.numpy())
                action_buffer.append(eval_action)
                distance_buffer.append(eval_distance)
                agent.eval_flg = False

            # check the stopping condition
            if global_timestep.numpy() > agent.params.num_frames:
                print("=== Training is Done ===")
                eval_reward, eval_distance, eval_action = eval_Agent_DDPG(env, agent)
                eval_epochs.append(global_timestep.numpy())
                action_buffer.append(eval_action)
                distance_buffer.append(eval_distance)
                visualise_act_and_dist(np.array(eval_epochs), np.array(action_buffer), np.array(distance_buffer),
                                       file_dir=agent.params.plot_path)
                env.close()
                break
