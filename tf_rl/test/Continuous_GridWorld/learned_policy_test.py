import argparse
from tf_rl.common.random_process import OrnsteinUhlenbeckProcess
from tf_rl.common.utils import *
from tf_rl.agents.DDPG import DDPG
from tf_rl.common.networks import DDPG_Actor as Actor, DDPG_Critic as Critic
from tf_rl.env.continuous_gridworld.env import GridWorld

eager_setup()

parser = argparse.ArgumentParser()
parser.add_argument("--env_name", default="Ant-v2", type=str, help="Env title")
parser.add_argument("--train_flg", default="original", type=str, help="train flg: original or on-policy")
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
parser.add_argument("--mu", default=0.3, type=float, help="magnitude of randomness")
parser.add_argument("--sigma", default=0.2, type=float, help="magnitude of randomness")
parser.add_argument("--action_range", default=[-1., 1.], type=list, help="magnitude of L2 regularisation")
parser.add_argument("--debug_flg", default=False, type=bool, help="debug mode or not")
parser.add_argument("--google_colab", default=False, type=bool, help="if you are executing this on GoogleColab")
params = parser.parse_args()
params.test_episodes = 1

now = datetime.datetime.now()

params.actor_model_dir = "../../logs/models/20190818-221419-DDPG_actor/"
params.critic_model_dir = "../../logs/models/20190818-221419-DDPG_critic/"

dense_goals = [(13.0, 8.0), (18.0, 11.0), (20.0, 15.0), (22.0, 19.0)]
env = GridWorld(max_episode_len=500, num_rooms=1, action_limit_max=1.0, silent_mode=False,
                start_position=(8.0, 8.0), goal_position=(22.0, 22.0), goal_reward=+100.0,
                dense_goals=dense_goals, dense_reward=+5,
                grid_len=30)

# set seed
env.seed(params.seed)
tf.random.set_random_seed(params.seed)

random_process = OrnsteinUhlenbeckProcess(size=env.action_space.shape[0], theta=0.15, mu=params.mu, sigma=params.sigma)
agent = DDPG(Actor, Critic, env.action_space.shape[0], random_process, params)
global_timestep = tf.compat.v1.train.get_or_create_global_step()

for ep in range(10):
    state = env.reset()
    done = False
    episode_reward = 0
    while not done:
        action = agent.eval_predict(state)
        next_state, reward, done, info = env.step(np.clip(action, -1.0, 1.0))
        state = next_state
        episode_reward += reward
    print("Ep Reward: ", episode_reward)