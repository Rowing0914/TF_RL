import gym, time
import argparse
import tensorflow as tf
from collections import deque
from tf_rl.common.monitor import Monitor
from tf_rl.common.random_process import OrnsteinUhlenbeckProcess, GaussianNoise
from tf_rl.common.memory import ReplayBuffer
from tf_rl.common.utils import *
from tf_rl.agents.DDPG import Graph_DDPG as DDPG
from tf_rl.common.networks import DDPG_Critic as Critic
from tf_rl.common.visualise import visualise_act_and_dist

from graph_nets import modules
from graph_nets.my_utils.input_graph_generator import GraphOperator
from graph_nets.my_utils.mujoco_parser import parse_mujoco_graph
from examples.DDPG.graph_nn_wrapper import ActorWrapper

eager_setup()

parser = argparse.ArgumentParser()
parser.add_argument("--env_name", default="CentipedeThree-v1", type=str, help="Env title")
parser.add_argument("--seed", default=123, type=int, help="seed for randomness")
# parser.add_argument("--num_frames", default=1_000_000, type=int, help="total frame in a training")
parser.add_argument("--num_frames", default=500_000, type=int, help="total frame in a training")
# parser.add_argument("--num_frames", default=40_000, type=int, help="total frame in a training")
parser.add_argument("--eval_interval", default=100_000, type=int, help="a frequency of evaluation in training phase")
# parser.add_argument("--eval_interval", default=10_000, type=int, help="a frequency of evaluation in training phase")
parser.add_argument("--memory_size", default=100_000, type=int, help="memory size in a training")
parser.add_argument("--learning_start", default=10_000, type=int, help="length before training")
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

# params.log_dir = "../../logs/logs/" + now.strftime("%Y%m%d-%H%M%S") + "-DDPG/"
# params.actor_model_dir = "../../logs/models/" + now.strftime("%Y%m%d-%H%M%S") + "-DDPG_actor/"
# params.critic_model_dir = "../../logs/models/" + now.strftime("%Y%m%d-%H%M%S") + "-DDPG_critic/"
# params.video_dir = "../../logs/video/video_{}".format(now.strftime("%Y%m%d-%H%M%S") + "_" + str(params.env_name))
# params.plot_path = "../../logs/plots/plot_{}/".format(now.strftime("%Y%m%d-%H%M%S") + "_" + str(params.env_name))
mu = str(params.mu).split(".")
mu = str(mu[0]+mu[1])
params.log_dir = "../../logs/logs/DDPG/{}-mu{}".format(str(params.env_name.split("-")[0]), mu)
params.actor_model_dir = "../../logs/models/DDPG/{}/actor-mu{}/".format(str(params.env_name.split("-")[0]), mu)
params.critic_model_dir = "../../logs/models/DDPG/{}/critic-mu{}/".format(str(params.env_name.split("-")[0]), mu)
params.video_dir = "../../logs/video/{}-mu{}".format(str(params.env_name.split("-")[0]), mu)
params.plot_path = "../../logs/plots/{}-mu{}/".format(str(params.env_name.split("-")[0]), mu)

env = gym.make(params.env_name)
env = Monitor(env, params.video_dir, force=True)

# set seed
env.seed(params.seed)
tf.compat.v1.random.set_random_seed(params.seed)

replay_buffer = ReplayBuffer(params.memory_size)
reward_buffer = deque(maxlen=params.reward_buffer_ep)
summary_writer = tf.contrib.summary.create_file_writer(params.log_dir)
# random_process = OrnsteinUhlenbeckProcess(size=env.action_space.shape[0], theta=0.15, mu=0.9, sigma=0.05)
random_process = GaussianNoise(mu=params.mu, sigma=params.sigma)

# instantiate the GraphNN
node_info = parse_mujoco_graph(task_name=params.env_name)  # get node_info
Graph_Operator = GraphOperator(input_dict=node_info["input_dict"],
                               output_list=node_info["output_list"],
                               obs_shape=node_info["debug_info"]["ob_size"])

env = gym.make(params.env_name)

# create model
graph_network = modules.GraphNetworkEager(internal_timestep=3,
                                          edge_layer_sizes=[16, 16, 1],
                                          node_layer_sizes=[16, 16, 1],
                                          global_layer_sizes=[16, 16, 1])

Actor = ActorWrapper(model=graph_network, graph_operator=Graph_Operator)

agent = DDPG(Actor, Critic, env.action_space.shape[0], random_process, params)

get_ready(agent.params)

global_timestep = tf.compat.v1.train.get_or_create_global_step()
time_buffer = deque(maxlen=agent.params.reward_buffer_ep)
log = logger(agent.params)
action_buffer, distance_buffer, eval_epochs = list(), list(), list()

with summary_writer.as_default():
    # for summary purpose, we put all codes in this context
    with tf.contrib.summary.always_record_summaries():

        for i in itertools.count():
            state = env.reset()
            total_reward = 0
            start = time.time()
            agent.random_process.reset_states()
            done = False
            episode_len = 0
            while not done:
                action = agent.predict(state).numpy()
                # scale for execution in env (in DDPG, every action is clipped between [-1, 1] in agent.predict)
                next_state, reward, done, info = env.step(action * env.action_space.high)
                replay_buffer.add(state, action, reward, next_state, done)

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

            # train the model at this point
            for t_train in range(int(episode_len / 10)):
                # for t_train in range(10): # for test purpose
                states, actions, rewards, next_states, dones = replay_buffer.sample(agent.params.batch_size)
                loss = agent.update(states, actions, rewards, next_states, dones)
                soft_target_model_update_eager(agent.target_critic, agent.critic, tau=agent.params.soft_update_tau)

            # save the update models
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
                eval_reward, eval_distance, eval_action = test_Agent_DDPG(agent)
                eval_epochs.append(global_timestep.numpy())
                action_buffer.append(eval_action)
                distance_buffer.append(eval_distance)
                agent.eval_flg = False

            # check the stopping condition
            if global_timestep.numpy() > agent.params.num_frames:
                print("=== Training is Done ===")
                eval_reward, eval_distance, eval_action = test_Agent_DDPG(agent)
                eval_epochs.append(global_timestep.numpy())
                action_buffer.append(eval_action)
                distance_buffer.append(eval_distance)
                visualise_act_and_dist(np.array(eval_epochs), np.array(action_buffer), np.array(distance_buffer),
                                       file_dir=agent.params.plot_path)
                env.close()
                break
