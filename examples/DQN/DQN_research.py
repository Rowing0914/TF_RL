import argparse, time
import tensorflow as tf
from collections import deque
from tf_rl.common.memory import ReplayBuffer
from tf_rl.common.policy import EpsilonGreedyPolicy_eager
from tf_rl.common.utils import *
from tf_rl.common.networks import CartPole as Model
from tf_rl.agents.DQN import DQN_cartpole

eager_setup()

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="CartPole", help="game env type: Atari or CartPole")
parser.add_argument("--seed", default=123, help="seed of randomness")
parser.add_argument("--loss_fn", default="huber", help="types of loss function: MSE or huber")
parser.add_argument("--grad_clip_flg", default="",
                    help="gradient clippings: by value(by_value) or global norm(norm) or nothing")
parser.add_argument("--num_frames", default=2000, type=int, help="total frame in a training")
parser.add_argument("--train_interval", default=1, type=int, help="a frequency of training occurring in training phase")
parser.add_argument("--eval_interval", default=2500, type=int,
                    help="a frequency of evaluation occurring in training phase")  # temp
parser.add_argument("--memory_size", default=5000, type=int, help="memory size in a training")
parser.add_argument("--learning_start", default=100, type=int,
                    help="frame number which specifies when to start updating the agent")
parser.add_argument("--sync_freq", default=1000, type=int, help="frequency of updating a target model")
parser.add_argument("--batch_size", default=32, type=int, help="batch size of each iteration of update")
parser.add_argument("--reward_buffer_ep", default=10, type=int, help="reward_buffer size")
parser.add_argument("--gamma", default=0.99, type=float,
                    help="discount factor: gamma > 1.0 or negative => does not converge!!")
parser.add_argument("--tau", default=1e-2, type=float, help="soft update tau")
parser.add_argument("--ep_start", default=1.0, type=float, help="initial value of epsilon")
parser.add_argument("--ep_end", default=0.02, type=float, help="final value of epsilon")
parser.add_argument("--lr_start", default=0.0025, type=float, help="initial value of lr")
parser.add_argument("--lr_end", default=0.00025, type=float, help="final value of lr")
parser.add_argument("--decay_steps", default=3000, type=int, help="a period for annealing a value(epsilon or beta)")
parser.add_argument("--debug_flg", default=False, type=bool, help="debug mode or not")
parser.add_argument("--google_colab", default=False, type=bool, help="if you are executing this on GoogleColab")
params = parser.parse_args()
params.goal = 195
params.test_episodes = 20

# init global time-step
global_timestep = tf.train.get_or_create_global_step()

# instantiate annealing funcs for ep and lr
anneal_ep = tf.train.polynomial_decay(params.ep_start, global_timestep, params.decay_steps, params.ep_end)
anneal_lr = tf.train.polynomial_decay(params.lr_start, global_timestep, params.decay_steps, params.lr_end)

# prep for training
policy = EpsilonGreedyPolicy_eager(Epsilon_fn=anneal_ep)
optimizer = tf.train.RMSPropOptimizer(anneal_lr, 0.99, 0.0, 1e-6)
replay_buffer = ReplayBuffer(params.memory_size)
reward_buffer = deque(maxlen=params.reward_buffer_ep)
loss_fn = create_loss_func(params.loss_fn)
grad_clip_fn = gradient_clip_fn(flag=params.grad_clip_flg)

# create a directory for log/model
params = create_log_model_directory(params, get_alg_name())
summary_writer = tf.contrib.summary.create_file_writer(params.log_dir)

# choose env and instantiate the agent correspondingly
env = MyWrapper(gym.make("CartPole-v0"))
agent = DQN_cartpole(Model, optimizer, loss_fn, grad_clip_fn, env.action_space.n, params)

# set seed
env.seed(params.seed)
tf.random.set_random_seed(params.seed)

get_ready(agent.params)
time_buffer = list()
global_timestep = tf.train.get_or_create_global_step()
log = logger(agent.params)
with summary_writer.as_default():
    # for summary purpose, we put all codes in this context
    with tf.contrib.summary.always_record_summaries():

        for i in itertools.count():
            state = env.reset()
            total_reward = 0
            start = time.time()
            cnt_action = list()
            done = False
            stds, means = [], []
            while not done:
                action = policy.select_action(agent, state)
                q = agent.predict(state)
                stds.append(np.std(q))
                means.append(np.mean(q))

                next_state, reward, done, info = env.step(action)
                replay_buffer.add(state, action, reward, next_state, done)

                global_timestep.assign_add(1)
                total_reward += reward
                state = next_state
                cnt_action.append(action)

                # for evaluation purpose
                if global_timestep.numpy() % agent.params.eval_interval == 0:
                    agent.eval_flg = True

                if (global_timestep.numpy() > agent.params.learning_start) and (
                        global_timestep.numpy() % agent.params.train_interval == 0):
                    states, actions, rewards, next_states, dones = replay_buffer.sample(agent.params.batch_size)

                    loss, batch_loss = agent.update(states, actions, rewards, next_states, dones)

                # synchronise the target and main models by hard
                if (global_timestep.numpy() > agent.params.learning_start) and (
                        global_timestep.numpy() % agent.params.sync_freq == 0):
                    agent.manager.save()
                    agent.target_model.set_weights(agent.main_model.get_weights())

            """
            ===== After 1 Episode is Done =====
            """

            tf.contrib.summary.scalar("reward", total_reward, step=i)
            tf.contrib.summary.scalar("exec time", time.time() - start, step=i)
            if i >= agent.params.reward_buffer_ep:
                tf.contrib.summary.scalar("Moving Ave Reward", np.mean(reward_buffer), step=i)
            tf.contrib.summary.histogram("taken actions", cnt_action, step=i)

            # store the episode reward
            reward_buffer.append(total_reward)
            time_buffer.append(time.time() - start)

            if global_timestep.numpy() > agent.params.learning_start and i % agent.params.reward_buffer_ep == 0:
                log.logging(global_timestep.numpy(), i, np.sum(time_buffer), reward_buffer, np.mean(loss),
                            policy.current_epsilon(), cnt_action)
                time_buffer = list()

            test_Agent(agent, env)
            print("Estimated Final Score Distribution: Std: {}, Mean: {}".format(np.sum(stds), np.mean(means)))

            if agent.eval_flg:
                # test_Agent(agent, env)
                agent.eval_flg = False

            # check the stopping condition
            if global_timestep.numpy() > agent.params.num_frames:
                print("=== Training is Done ===")
                test_Agent(agent, env, n_trial=agent.params.test_episodes)
                print("Estimated Final Score Distribution: Std: {}, Mean: {}".format(np.sum(stds), np.mean(means)))
                env.close()
                break
