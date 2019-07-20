import gym
from gym_extensions.continuous import mujoco
from gym.wrappers import Monitor
import argparse
import tensorflow as tf
from tf_rl.common.utils import eager_setup
from tf_rl.agents.DDPG import DDPG
from tf_rl.common.networks import DDPG_Actor as Actor, DDPG_Critic as Critic

eager_setup()

parser = argparse.ArgumentParser()
parser.add_argument("--env_name", default="Humanoid-v2", help="Env title")
parser.add_argument("--seed", default=123, type=int, help="seed for randomness")
parser.add_argument("--num_frames", default=1_000_000, type=int, help="total frame in a training")
parser.add_argument("--train_interval", default=100, type=int, help="a frequency of training in training phase")
parser.add_argument("--nb_train_steps", default=50, type=int, help="a number of training after one episode")
parser.add_argument("--eval_interval", default=50_000, type=int, help="a frequency of evaluation in training phase")
parser.add_argument("--memory_size", default=100_000, type=int, help="memory size in a training")
parser.add_argument("--learning_start", default=10_000, type=int, help="length before training")
parser.add_argument("--batch_size", default=100, type=int, help="batch size of each iteration of update")
parser.add_argument("--reward_buffer_ep", default=10, type=int, help="reward_buffer size")
parser.add_argument("--gamma", default=0.99, type=float, help="discount factor")
parser.add_argument("--soft_update_tau", default=1e-2, type=float, help="soft-update tau ")
parser.add_argument("--L2_reg", default=0.5, type=float, help="magnitude of L2 regularisation")
parser.add_argument("--action_range", default=[-1., 1.], type=list, help="magnitude of L2 regularisation")
parser.add_argument("--debug_flg", default=False, type=bool, help="debug mode or not")
parser.add_argument("--model_name", default="mu00", type=str, help="path to pre-trained model")
parser.add_argument("--google_colab", default=False, type=bool, help="if you are executing this on GoogleColab")
params = parser.parse_args()
params.test_episodes = 10
params.goal = 0
params.actor_model_dir = "./models/actor_{}/".format(params.model_name)
params.critic_model_dir = "./models/critic_{}/".format(params.model_name)

# available env list: https://github.com/Rowing0914/gym-extensions/blob/mujoco200/tests/all_tests.py
HalfCheetah_Env_list = [
    "HalfCheetahGravityHalf-v1",
    "HalfCheetahGravityThreeQuarters-v1",
    "HalfCheetahGravityOneAndHalf-v1",
    "HalfCheetahGravityOneAndQuarter-v1",
    # "HalfCheetahWall-v1",       => this uses different obs shape since it includes the sensors
    # "HalfCheetahWithSensor-v1", => this uses different obs shape since it includes the sensors
    "HalfCheetahBigTorso-v1",
    "HalfCheetahBigThigh-v1",
    "HalfCheetahBigLeg-v1",
    "HalfCheetahBigFoot-v1",
    "HalfCheetahSmallTorso-v1",
    "HalfCheetahSmallThigh-v1",
    "HalfCheetahSmallLeg-v1",
    "HalfCheetahSmallFoot-v1",
    "HalfCheetahSmallHead-v1",
    "HalfCheetahBigHead-v1"
]

tf.random.set_random_seed(params.seed)
random_process = None
agent = DDPG(Actor, Critic, 6, random_process, params)

for env_name in HalfCheetah_Env_list:
    print(env_name)
    env = gym.make(env_name)
    env = Monitor(env, "./video/{}/video_{}".format(params.model_name, env_name), force=True)

    # set seed
    env.seed(params.seed)

    n_trial = 1
    all_rewards = list()
    for ep in range(n_trial):
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            # env.render()
            action = agent.eval_predict(state)
            # scale for execution in env (in DDPG, every action is clipped between [-1, 1] in agent.predict)
            next_state, reward, done, _ = env.step(action * env.action_space.high)
            state = next_state
            episode_reward += reward

        all_rewards.append(episode_reward)
        print("| Ep: {}/{} | Score: {} |".format(ep + 1, n_trial, episode_reward))
    env.close()
