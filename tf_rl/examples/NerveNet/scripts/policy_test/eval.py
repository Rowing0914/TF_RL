import argparse
from tf_rl.common.monitor import Monitor
from tf_rl.common.random_process import OrnsteinUhlenbeckProcess, GaussianNoise
from tf_rl.common.utils import *
from tf_rl.agents.DDPG import DDPG
from tf_rl.common.networks import DDPG_Actor as Actor, DDPG_Critic as Critic
import environments.register as register

eager_setup()

parser = argparse.ArgumentParser()
parser.add_argument("--env_name", default="AntS-v1", type=str, help="Env title")
parser.add_argument("--seed", default=123, type=int, help="seed for randomness")
parser.add_argument("--eval_interval", default=100_000, type=int, help="a frequency of evaluation in training phase")
parser.add_argument("--mu", default=0.2, type=float, help="magnitude of randomness")
parser.add_argument("--sigma", default=0.05, type=float, help="magnitude of randomness")
parser.add_argument("--n_trial", default=10, type=int, help="num of eval ep")
parser.add_argument("--action_range", default=[-1., 1.], type=list, help="magnitude of L2 regularisation")
params = parser.parse_args()

params.actor_model_dir = "../../logs/models/20190731-181814-DDPG-GGNN_actor/"
params.critic_model_dir = "../../logs/models/20190731-181814-DDPG-GGNN_critic/"
params.video_dir = "./video_{}".format(str(params.env_name))

env = gym.make(params.env_name)
env = Monitor(env, params.video_dir, force=True)

random_process = GaussianNoise(mu=0.0, sigma=0.0)
agent = DDPG(Actor, Critic, env.action_space.shape[0], random_process, params)

global_timestep = tf.compat.v1.train.get_or_create_global_step()

all_distances, all_rewards, all_actions = list(), list(), list()
distance_func = get_distance(agent.params.env_name) # create the distance measure func
print("=== Evaluation Mode ===")
for ep in range(params.n_trial):
    env.record_start()
    obs = env.reset()
    state = obs["flat_obs"]
    done = False
    episode_reward = 0
    while not done:
        action = agent.eval_predict(state)
        # action = env.action_space.sample()

        # scale for execution in env (in DDPG, every action is clipped between [-1, 1] in agent.predict)
        obs, reward, done, info = env.step(action * env.action_space.high)
        # print(action, reward)
        next_flat_state, next_graph_state = obs["flat_obs"], obs["graph_obs"]
        distance = distance_func(action, reward, info)
        all_actions.append(action.mean()**2) # Mean Squared of action values
        all_distances.append(distance)
        state = next_flat_state
        episode_reward += reward

    all_rewards.append(episode_reward)
    tf.contrib.summary.scalar("Evaluation Score", episode_reward, step=agent.index_timestep)
    print("| Ep: {}/{} | Score: {} |".format(ep + 1, params.n_trial, episode_reward))
env.record_end()