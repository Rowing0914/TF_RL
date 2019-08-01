import tensorflow as tf
import numpy as np
import os, datetime, itertools, shutil, gym, sys
from tf_rl.common.visualise import plot_Q_values
from tf_rl.common.wrappers import MyWrapper, CartPole_Pixel, wrap_deepmind, make_atari

"""

TF basic Utility functions

"""


def eager_setup():
    """
    it eables an eager execution in tensorflow with config that allows us to flexibly access to a GPU
    from multiple python scripts

    :return:
    """
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True,
                                      intra_op_parallelism_threads=1,
                                      inter_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True
    tf.compat.v1.enable_eager_execution(config=config)
    tf.compat.v1.enable_resource_variables()


"""

Common Utility functions 

"""


def get_alg_name():
    """Returns the name of the algorithm.
    We assume that the directory architecutre for that algo looks like below
        - Atari: `examples/algo_name/algo_name_eager.py`
        - Cartpole: `examples/algo_name/algo_name_eager_cartpole.py`
        * where algo_name must be uppercase/capital letters!!
    """
    alg_name = sys.argv[0].rsplit("/")[-1].rsplit(".")[0].replace("_eager", "")
    return alg_name


def invoke_agent_env(params, alg):
    """Returns the wrapped env and string name of agent, then Use `eval(agent)` to activate it from main script
    """
    if params.mode == "Atari":
        env = wrap_deepmind(make_atari("{}NoFrameskip-v4".format(params.env_name, skip_frame_k=params.skip_frame_k)),
                            skip_frame_k=params.skip_frame_k)
        if params.debug_flg:
            agent = "{}_debug".format(alg)
        else:
            agent = "{}".format(alg)
    else:
        agent = "{}".format(alg)
        if params.mode == "CartPole":
            env = MyWrapper(gym.make("CartPole-v0"))
        elif params.mode == "CartPole-p":
            env = CartPole_Pixel(gym.make("CartPole-v0"))
    return agent, env


def create_log_model_directory(params, alg):
    """
    Create a directory for log/model
    this is compatible with Google colab and can connect to MyDrive through the authorisation step

    :param params:
    :return:
    """
    if params.mode in ["Atari", "atari", "MuJoCo", "mujoco"]:
        second_name = params.env_name
    else:
        second_name = params.mode
    now = datetime.datetime.now()

    if params.google_colab:
        # mount the MyDrive on google drive and create the log directory for saving model and logging using tensorboard
        params.log_dir, params.model_dir, params.log_dir_colab, params.model_dir_colab = _setup_on_colab(alg,
                                                                                                         params.mode)
    else:
        if params.debug_flg:
            params.log_dir = "../../logs/logs/" + now.strftime("%Y%m%d-%H%M%S") + "-{}_{}_debug/".format(alg,
                                                                                                         second_name)
            params.model_dir = "../../logs/models/" + now.strftime("%Y%m%d-%H%M%S") + "-{}_{}_debug/".format(alg,
                                                                                                             second_name)
        else:
            params.log_dir = "../../logs/logs/" + now.strftime("%Y%m%d-%H%M%S") + "-{}_{}/".format(alg, second_name)
            params.model_dir = "../../logs/models/" + now.strftime("%Y%m%d-%H%M%S") + "-{}_{}/".format(alg, second_name)
    return params


def create_loss_func(loss_name="mse"):
    if loss_name == "huber":
        loss_fn = tf.compat.v1.losses.huber_loss
    elif loss_name == "mse":
        loss_fn = tf.compat.v1.losses.mean_squared_error
    else:
        assert False, "Choose the loss_fn from either huber or mse"
    return loss_fn


def get_ready(params):
    """
    Print out the content of params

    :param params:
    :return:
    """
    for key, item in vars(params).items():
        print(key, " : ", item)


def create_checkpoint(model, optimizer, model_dir):
    """
    Create a checkpoint for managing a model

    :param model:
    :param optimizer:
    :param model_dir:
    :return:
    """
    checkpoint_dir = model_dir
    check_point = tf.train.Checkpoint(optimizer=optimizer,
                                      model=model,
                                      optimizer_step=tf.compat.v1.train.get_or_create_global_step())
    manager = tf.train.CheckpointManager(check_point, checkpoint_dir, max_to_keep=3)

    # try re-loading the previous training progress!
    try:
        print("Try loading the previous training progress")
        check_point.restore(manager.latest_checkpoint)
        assert tf.compat.v1.train.get_global_step().numpy() != 0
        print("===================================================\n")
        print("Restored the model from {}".format(checkpoint_dir))
        print("Currently we are on time-step: {}".format(tf.compat.v1.train.get_global_step().numpy()))
        print("\n===================================================")
    except:
        print("===================================================\n")
        print("Previous Training files are not found in Directory: {}".format(checkpoint_dir))
        print("\n===================================================")
    return manager


def _setup_on_colab(alg_name, env_name):
    """
    Mount MyDrive to current instance through authentication of Google account
    Then use it as a backup of training related files

    :param env_name:
    :return:
    """
    # mount your drive on google colab
    from google.colab import drive
    drive.mount("/content/gdrive")
    log_dir = "/content/TF_RL/logs/logs/{}/{}".format(alg_name, env_name)
    model_dir = "/content/TF_RL/logs/models/{}/{}".format(alg_name, env_name)
    log_dir_colab = "/content/gdrive/My Drive/logs/logs/{}/{}".format(alg_name, env_name)
    model_dir_colab = "/content/gdrive/My Drive/logs/models/{}/{}".format(alg_name, env_name)

    # create the logs directory under the root dir
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    # if the previous directory existed in My Drive, then we would continue training on top of the previous training
    if os.path.isdir(log_dir_colab):
        print("=== {} IS FOUND ===".format(log_dir_colab))
        copy_dir(log_dir_colab, log_dir, verbose=True)
    else:
        print("=== {} IS NOT FOUND ===".format(log_dir_colab))
        os.makedirs(log_dir_colab)
        print("=== FINISHED CREATING THE DIRECTORY ===")

    if os.path.isdir(model_dir_colab):
        print("=== {} IS FOUND ===".format(model_dir_colab))
        copy_dir(model_dir_colab, model_dir, verbose=True)
    else:
        print("=== {} IS NOT FOUND ===".format(model_dir_colab))
        os.makedirs(model_dir_colab)
        print("=== FINISHED CREATING THE DIRECTORY ===")
    return log_dir, model_dir, log_dir_colab, model_dir_colab


class AnnealingSchedule:
    """
    Scheduling the gradually decreasing value, e.g., epsilon or beta params

    """

    def __init__(self, start=1.0, end=0.1, decay_steps=500, decay_type="linear"):
        self.start = start
        self.end = end
        self.decay_steps = decay_steps
        self.annealed_value = np.linspace(start, end, decay_steps)
        self.decay_type = decay_type

    def old_get_value(self, timestep):
        """
        Deprecated

        :param timestep:
        :return:
        """
        if self.decay_type == "linear":
            return self.annealed_value[min(timestep, self.decay_steps) - 1]
        # don't use this!!
        elif self.decay_type == "curved":
            if timestep < self.decay_steps:
                return self.start * 0.9 ** (timestep / self.decay_steps)
            else:
                return self.end

    def get_value(self):
        timestep = tf.train.get_or_create_global_step()  # we are maintaining the global-step in train.py so it is accessible
        if self.decay_type == "linear":
            return self.annealed_value[min(timestep.numpy(), self.decay_steps) - 1]
        # don't use this!!
        elif self.decay_type == "curved":
            if timestep.numpy() < self.decay_steps:
                return self.start * 0.9 ** (timestep.numpy() / self.decay_steps)
            else:
                return self.end


def copy_dir(src, dst, symlinks=False, ignore=None, verbose=False):
    """
    copy the all contents in `src` directory to `dst` directory

    Usage:
        ```python
        delete_files("./bb/")
        ```
    """
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if verbose:
            print("From:{}, To: {}".format(s, d))
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


def delete_files(folder, verbose=False):
    """
    delete the all contents in `folder` directory

    Usage:
        ```python
        copy_dir("./aa/", "./bb/")
        ```
    """
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
                if verbose:
                    print("{} has been deleted".format(file_path))
        except Exception as e:
            print(e)


class RunningMeanStd:
    """
    Running Mean and Standard Deviation for normalising the observation!
    This is mainly used in MuJoCo experiments, e.g. DDPG!

    Formula:
        - Normalisation: y = (x-mean)/std
    """

    def __init__(self, shape, clip_range=5, epsilon=1e-2):
        self.size = shape
        self.epsilon = epsilon
        self.clip_range = clip_range
        self._sum = 0.0
        self._sumsq = np.ones(self.size, np.float32) * epsilon
        self._count = np.ones(self.size, np.float32) * epsilon
        self.mean = self._sum / self._count
        self.std = np.sqrt(np.maximum(self._sumsq / self._count - np.square(self.mean), np.square(self.epsilon)))

    def update(self, x):
        """
        update the mean and std by given input

        :param x: can be observation, reward, or action!!
        :return:
        """
        x = x.reshape(-1, self.size)
        self._sum = x.sum(axis=0)
        self._sumsq = np.square(x).sum(axis=0)
        self._count = np.array([len(x)], dtype='float64')

        self.mean = self._sum / self._count
        self.std = np.sqrt(np.maximum(self._sumsq / self._count - np.square(self.mean), np.square(self.epsilon)))

    def normalise(self, x):
        """
        Using well-maintained mean and std, we normalise the input followed by update them.

        :param x:
        :return:
        """
        result = np.clip((x - self.mean) / self.std, -self.clip_range, self.clip_range)
        return result


def test(sess, agent, env, params):
    xmax = agent.num_action
    ymax = 3

    print("\n ===== TEST STARTS: {0} Episodes =====  \n".format(params.test_episodes))

    for i in range(params.test_episodes):
        state = env.reset()
        for t in itertools.count():
            env.render()
            q_values = sess.run(agent.pred, feed_dict={agent.state: state.reshape(params.state_reshape)})[0]
            action = np.argmax(q_values)
            plot_Q_values(q_values, xmax=xmax, ymax=ymax)
            obs, reward, done, _ = env.step(action)
            state = obs
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
    return


class logger:
    def __init__(self, params):
        self.params = params
        self.prev_update_step = 0

    def logging(self, time_step, current_episode, exec_time, reward_buffer, loss, epsilon, cnt_action):
        """
        Logging function

        :param time_step:
        :param max_steps:
        :param current_episode:
        :param exec_time:
        :param reward:
        :param loss:
        :param cnt_action:
        :return:
        """
        cnt_actions = dict((x, cnt_action.count(x)) for x in set(cnt_action))
        episode_steps = time_step - self.prev_update_step
        # remaing_time_step/exec_time_for_one_step
        remaining_time = str(datetime.timedelta(
            seconds=(self.params.num_frames - time_step) * exec_time / (episode_steps)))
        print(
            "{0}/{1}: Ep: {2}({3:.1f} fps), Remaining: {4}, (R) GOAL: {5}, {6} Ep => [MEAN: {7:.3f}, MAX: {8:.3f}], (last ep) Loss: {9:.3f}, Eps: {10:.3f}, Act: {11}".format(
                time_step, self.params.num_frames, current_episode, episode_steps / exec_time, remaining_time,
                self.params.goal, self.params.reward_buffer_ep, np.mean(reward_buffer), np.max(reward_buffer), loss,
                epsilon, cnt_actions
            ))
        self.prev_update_step = time_step


"""
Algorithm Specific Utility functions

"""


class her_sampler:
    # borrow from: https://github.com/TianhongDai/hindsight-experience-replay/blob/master/her.py
    def __init__(self, replay_strategy, replay_k, reward_func=None):
        self.replay_strategy = replay_strategy
        self.replay_k = replay_k
        if self.replay_strategy == 'future':
            self.future_p = 1 - (1. / (1 + replay_k))
        else:
            self.future_p = 0
        self.reward_func = reward_func

    def sample_her_transitions(self, episode_batch, batch_size_in_transitions):
        T = episode_batch['actions'].shape[1]
        rollout_batch_size = episode_batch['actions'].shape[0]
        batch_size = batch_size_in_transitions
        # select which rollouts and which timesteps to be used
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}
        # her idx
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]
        # replace go with achieved goal
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag
        # to get the params to re-compute reward
        transitions['r'] = np.expand_dims(self.reward_func(transitions['ag_next'], transitions['g'], None), 1)
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}

        return transitions


def action_postprocessing(action, params):
    action += params.noise_eps * params.max_action * np.random.randn(*action.shape)
    action = np.clip(action, -params.max_action, params.max_action)
    # random actions...
    random_actions = np.random.uniform(low=-params.max_action,
                                       high=params.max_action,
                                       size=params.num_action)
    # choose if use the random actions
    action += np.random.binomial(1, params.random_eps, 1)[0] * (random_actions - action)
    return action


def state_unpacker(state):
    """
    Given the dictionary of state, it unpacks and returns processed items as numpy.ndarray

    Sample input:
        {'observation': array([ 1.34193265e+00,  7.49100375e-01,  5.34722720e-01,  1.30179339e+00, 8.86399624e-01,
                                4.24702091e-01, -4.01392554e-02,  1.37299250e-01, -1.10020629e-01,  2.91834773e-06,
                                -4.72661656e-08, -3.85214084e-07, 5.92637053e-07,  1.12208536e-13, -7.74656889e-06,
                                -7.65027248e-08, 4.92570535e-05,  1.88857148e-07, -2.90549459e-07, -1.18156686e-18,
                                7.73934983e-06,  7.18103404e-08, -2.42928780e-06,  4.93607091e-07, 1.70999820e-07]),
        'achieved_goal': array([1.30179339, 0.88639962, 0.42470209]),
        'desired_goal': array([1.4018907 , 0.62021174, 0.4429846 ])}

    :param state:
    :return:
    """
    obs = np.array(state["observation"])
    achieved_goal = np.array(state["achieved_goal"])
    desired_goal = np.array(state["desired_goal"])
    remaining_goal = simple_goal_subtract(desired_goal, achieved_goal)
    return obs, achieved_goal, desired_goal, remaining_goal


def simple_goal_subtract(goal, achieved_goal):
    """
    We subtract the achieved goal from the desired one to see how much we are still far from the desired position
    """
    assert goal.shape == achieved_goal.shape
    return goal - achieved_goal


ALIVE_BONUS = 1.0


def get_distance(env_name):
    """
    This returns the distance according to the implementation of env
    For instance, halfcheetah and humanoid have the different way to return the distance
    so that we need to deal with them accordingly.
    :return: func to calculate the distance(float)
    """
    obj_name = env_name.split("-")[0]
    if not obj_name.find("Ant") == -1:
        def func(action, reward, info):
            # https://github.com/openai/gym/blob/master/gym/envs/mujoco/ant.py#L14
            distance = info["reward_forward"]
            return distance
    elif not obj_name.find("HalfCheetah") == -1:
        def func(action, reward, info):
            # https://github.com/openai/gym/blob/master/gym/envs/mujoco/half_cheetah.py
            distance = info["reward_run"]
            return distance
    elif not obj_name.find("Hopper") == -1:
        def func(action, reward, info):
            # https://github.com/openai/gym/blob/master/gym/envs/mujoco/hopper.py#L15
            distance = (reward - ALIVE_BONUS) + 1e-3 * np.square(action).sum()
            return distance
    elif not obj_name.find("Humanoid") == -1:
        def func(action, reward, info):
            # https://github.com/openai/gym/blob/master/gym/envs/mujoco/humanoid.py#L30
            distance = info["reward_linvel"] / 1.25
            return distance
    elif not obj_name.find("Swimmer") == -1:
        def func(action, reward, info):
            # https://github.com/openai/gym/blob/master/gym/envs/mujoco/swimmer.py#L15
            distance = info["reward_fwd"]
            return distance
    elif not obj_name.find("Walker2d") == -1:
        def func(action, reward, info):
            # https://github.com/openai/gym/blob/master/gym/envs/mujoco/walker2d.py#L16 -> original version
            distance = (reward - ALIVE_BONUS) + 1e-3 * np.square(action).sum()

            # https://github.com/openai/gym/blob/master/gym/envs/mujoco/walker2d_v3.py#L90 -> version 3.0
            # distance = info["x_velocity"]
            return distance
    elif not obj_name.find("Centipede") == -1:
        def func(action, reward, info):
            distance = info["reward_forward"]
            return distance
    else:
        assert False, "This env: {} is not supported yet.".format(env_name)
    return func


"""
TODO: I think I will remove this.

# Copyright 2018 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
===== Tracker is A class for storing iteration-specific metrics. ====


"""


class Tracker(object):
    """A class for storing iteration-specific metrics.

    The internal format is as follows: we maintain a mapping from keys to lists.
    Each list contains all the values corresponding to the given key.

    For example, self.data_lists['train_episode_returns'] might contain the
      per-episode returns achieved during this iteration.

    Attributes:
      data_lists: dict mapping each metric_name (str) to a list of said metric
        across episodes.
    """

    def __init__(self):
        self.data_lists = {}

    def append(self, data_pairs):
        """Add the given values to their corresponding key-indexed lists.

        Args:
          data_pairs: A dictionary of key-value pairs to be recorded.
        """
        for key, value in data_pairs.items():
            if key not in self.data_lists:
                self.data_lists[key] = []
            self.data_lists[key].append(value)


"""

Update methods 

"""


def sync_main_target(sess, target, source):
    """
    Synchronise the models
    from Denny Britz's excellent RL repo
    https://github.com/dennybritz/reinforcement-learning/blob/master/DQN/Double%20DQN%20Solution.ipynb

    :param main:
    :param target:
    :return:
    """
    source_params = [t for t in tf.trainable_variables() if t.name.startswith(source.scope)]
    source_params = sorted(source_params, key=lambda v: v.name)
    target_params = [t for t in tf.trainable_variables() if t.name.startswith(target.scope)]
    target_params = sorted(target_params, key=lambda v: v.name)

    update_ops = []
    for target_w, source_w in zip(target_params, source_params):
        op = target_w.assign(source_w)
        update_ops.append(op)

    sess.run(update_ops)


def soft_target_model_update(sess, target, source, tau=1e-2):
    """
    Soft update model parameters.
    target = tau * source + (1 - tau) * target

    :param main:
    :param target:
    :param tau:
    :return:
    """
    source_params = [t for t in tf.trainable_variables() if t.name.startswith(source.scope)]
    source_params = sorted(source_params, key=lambda v: v.name)
    target_params = [t for t in tf.trainable_variables() if t.name.startswith(target.scope)]
    target_params = sorted(target_params, key=lambda v: v.name)

    update_ops = []
    for target_w, source_w in zip(target_params, source_params):
        # target = tau * source + (1 - tau) * target
        op = target_w.assign(tau * source_w + (1 - tau) * target_w)
        update_ops.append(op)

    sess.run(update_ops)


@tf.contrib.eager.defun(autograph=False)
def soft_target_model_update_eager(target, source, tau=1e-2):
    """
    Soft update model parameters.
    target = tau * source + (1 - tau) * target

    :param main:
    :param target:
    :param tau:
    :return:
    """

    for param, target_param in zip(source.weights, target.weights):
        target_param.assign(tau * param + (1 - tau) * target_param)


"""

Gradient Clipping

"""


def gradient_clip_fn(flag=None):
    """
    given a flag, create the clipping function and returns it as a function
    currently it supports:
        - by_value
        - norm
        - None

    :param flag:
    :return:
    """
    if flag == "":
        def _func(grads):
            return grads
    elif flag == "by_value":
        def _func(grads):
            grads = [ClipIfNotNone(grad, -1., 1.) for grad in grads]
            return grads
    elif flag == "norm":
        def _func(grads):
            grads, _ = tf.clip_by_global_norm(grads, 10.0)
            return grads
    else:
        assert False, "Choose the gradient clipping function from by_value, norm, or nothing!"
    return _func


def ClipIfNotNone(grad, _min, _max):
    """
    Reference: https://stackoverflow.com/a/39295309
    :param grad:
    :return:
    """
    if grad is None:
        return grad
    return tf.clip_by_value(grad, _min, _max)


"""

Test Methods

"""


def eval_Agent(agent, env, n_trial=1):
    """
    Evaluate the trained agent!

    :return:
    """
    all_rewards = list()
    print("=== Evaluation Mode ===")
    for ep in range(n_trial):
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            # epsilon-greedy for evaluation using a fixed epsilon of 0.05(Nature does this!)
            if np.random.uniform() < 0.05:
                action = np.random.randint(agent.num_action)
            else:
                action = np.argmax(agent.predict(state))
            next_state, reward, done, _ = env.step(action)
            state = next_state
            episode_reward += reward

        all_rewards.append(episode_reward)
        tf.contrib.summary.scalar("Evaluation Score", episode_reward, step=agent.index_timestep)
        print("| Ep: {}/{} | Score: {} |".format(ep + 1, n_trial, episode_reward))

    # if this is running on Google Colab, we would store the log/models to mounted MyDrive
    if agent.params.google_colab:
        delete_files(agent.params.model_dir_colab)
        delete_files(agent.params.log_dir_colab)
        copy_dir(agent.params.log_dir, agent.params.log_dir_colab)
        copy_dir(agent.params.model_dir, agent.params.model_dir_colab)

    if n_trial > 2:
        print("=== Evaluation Result ===")
        all_rewards = np.array([all_rewards])
        print("| Max: {} | Min: {} | STD: {} | MEAN: {} |".format(np.max(all_rewards), np.min(all_rewards),
                                                                  np.std(all_rewards), np.mean(all_rewards)))


def eval_Agent_DDPG(env, agent, n_trial=1):
    """
    Evaluate the trained agent with the recording of its behaviour

    :return:
    """

    all_distances, all_rewards, all_actions = list(), list(), list()
    distance_func = get_distance(agent.params.env_name)  # create the distance measure func
    print("=== Evaluation Mode ===")
    for ep in range(n_trial):
        env.record_start()
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = agent.eval_predict(state)
            # scale for execution in env (in DDPG, every action is clipped between [-1, 1] in agent.predict)
            next_state, reward, done, info = env.step(action * env.action_space.high)
            distance = distance_func(action, reward, info)
            all_actions.append(action.mean() ** 2)  # Mean Squared of action values
            all_distances.append(distance)
            state = next_state
            episode_reward += reward

        all_rewards.append(episode_reward)
        tf.contrib.summary.scalar("Evaluation Score", episode_reward, step=agent.index_timestep)
        print("| Ep: {}/{} | Score: {} |".format(ep + 1, n_trial, episode_reward))
    env.record_end()
    return all_rewards, all_distances, all_actions


def eval_Agent_TRPO(agent, env, n_trial=1):
    """
    Evaluate the trained agent!

    :return:
    """
    all_rewards = list()
    print("=== Evaluation Mode ===")
    for ep in range(n_trial):
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = agent.predict(state)
            # scale for execution in env (in DDPG, every action is clipped between [-1, 1] in agent.predict)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            episode_reward += reward

        all_rewards.append(episode_reward)
        tf.contrib.summary.scalar("Evaluation Score", episode_reward, step=agent.index_timestep)
        print("| Ep: {}/{} | Score: {} |".format(ep + 1, n_trial, episode_reward))

    if n_trial > 2:
        print("=== Evaluation Result ===")
        all_rewards = np.array([all_rewards])
        print("| Max: {} | Min: {} | STD: {} | MEAN: {} |".format(np.max(all_rewards), np.min(all_rewards),
                                                                  np.std(all_rewards), np.mean(all_rewards)))


def eval_Agent_HER(agent, env, n_trial=1):
    """
    Evaluate the trained agent!

    :return:
    """
    successes = list()
    for ep in range(n_trial):
        state = env.reset()
        # obs, achieved_goal, desired_goal in `numpy.ndarray`
        obs, ag, dg, rg = state_unpacker(state)
        success = list()
        for ts in range(agent.params.num_steps):
            # env.render()
            action = agent.predict(obs, dg)
            # action = action_postprocessing(action, agent.params)
            next_state, reward, done, info = env.step(action)
            success.append(info.get('is_success'))
            # obs, achieved_goal, desired_goal in `numpy.ndarray`
            next_obs, next_ag, next_dg, next_rg = state_unpacker(next_state)
            obs = next_obs
            dg = next_dg
        successes.append(success)
    return np.mean(np.array(successes))
