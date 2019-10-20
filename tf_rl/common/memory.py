"""
Design reference
https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
"""

import tensorflow as tf
import numpy as np
import shutil
import random
import json
import os

def check_path(path):
    if path[-1] == "/":
        return path[:-1]
    else:
        return path

class ReplayBuffer(object):
    """ Experience Replay Memory Buffer """

    def __init__(self, size,
                 n_step=0,
                 gamma=0.99,
                 flg_seq=True,
                 traj_dir="/tmp",
                 recover_data=False):
        self._storage = []
        self._maxsize = size
        self._n_step = n_step
        self._gamma = gamma
        self._next_idx = 0
        self._save_idx = 0
        self._flg_seq = flg_seq
        self.traj_dir = check_path(path=traj_dir)

        if recover_data:
            self.load_tf()

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def add_zero_transition(self, obs_t, action, reward, obs_tp1, done):
        """
        Takes as input all items involve in one time-step and
        adds a padding transition filled with zeros (Used in episode beginnings).
        """
        obs_t = np.zeros(obs_t.shape)
        action = np.zeros(action.shape)
        reward = np.zeros(reward.shape)
        obs_tp1 = np.zeros(obs_tp1.shape)
        done = np.zeros(done)
        self.add(obs_t, action, reward, obs_tp1, done)

    def _encode_sample(self, idxes):
        """
        One step sampling method
        """
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def _encode_sample_n_step_sequence(self, idxes):
        """
        n-consecutive time-step sampling method
        Return:
            obs, act, rew, next_obs, done FROM t to t+n
        """
        # Resulting arrays
        obs_t, action, reward, obs_tp1, done = [], [], [], [], []

        # === Sampling method ===
        for i in idxes:
            if i + self._n_step > len(self._storage) - 1:  # avoid the index out of range error!!
                first_half = len(self._storage) - 1 - i
                second_half = self._n_step - first_half
                data_n = self._storage[i: i + first_half] + self._storage[:second_half]
            else:
                data_n = self._storage[i: i + self._n_step]

            # TODO: this might not be efficient... because for every index we go through n-step
            # so that consider using separate bucket for each component(o, a, r, no, d)
            # then we can just specify the indices of them instead of having for-loop to operate on them.
            obs_t_n, action_n, reward_n, obs_tp1_n, done_n = [], [], [], [], []
            for _data in data_n:
                _o, _a, _r, _no, _d = _data
                obs_t_n.append(np.array(_o, copy=False))
                action_n.append(_a)
                reward_n.append(_r)
                obs_tp1_n.append(np.array(_no, copy=False))
                done_n.append(_d)

            # Store a data at each time-sequence in the resulting array
            obs_t.append(np.array(obs_t_n, copy=False))
            action.append(action_n)
            reward.append(reward_n)
            obs_tp1.append(np.array(obs_tp1_n, copy=False))
            done.append(done_n)
        return np.array(obs_t), np.array(action), np.array(reward), np.array(obs_tp1), np.array(done)

    def _encode_sample_n_step(self, idxes):
        """
        n-step ahead sampling method
        Return:
            obs, act, rew, next_obs, done at t as well as t+n
        """
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            if i + self._n_step > len(self._storage) - 1:  # avoid the index out of range error!!
                _idx = (i + self._n_step) - len(self._storage) + 1
                data_n = self._storage[i + self._n_step - _idx]
                reward_n = np.sum(
                    [self._gamma ** i * _data[2] for i, _data in enumerate(self._storage[i: i + self._n_step - _idx])])
            else:
                data_n = self._storage[i + self._n_step]
                # we need to compute the discounted reward for training
                reward_n = np.sum(
                    [self._gamma ** i * _data[2] for i, _data in enumerate(self._storage[i: i + self._n_step])])
            obs_t, action, reward, obs_tp1, done = data
            obs_t_n, action_n, _, obs_tp1_n, done_n = data_n
            obses_t.append(np.concatenate([obs_t, obs_t_n], axis=-1))
            actions.append(np.array([action, action_n]))
            rewards.append(np.array([reward, reward_n]))
            obses_tp1.append(np.concatenate([obs_tp1, obs_tp1_n], axis=-1))
            dones.append(np.array([done, done_n]))
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def _get_data_to_save(self):
        """
        Extract data in the storage to save it at each epoch
        Typical two pointers approach:
            - self._save_idx: slow pointer
            - self._next_idx: fast pointer

        <Example>
          _____
            |
            |<-- self._save_idx
            |==|
            |  |
            |  | upon call this func, we save this area in the storage
            |  |
            |==|
            |<-- self._next_idx
            |
          ˉˉˉˉˉ
        """
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in range(len(self._storage[self._save_idx:self._next_idx])):
            data = self._storage[self._save_idx + i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        self._save_idx = self._next_idx
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def _get_all_data(self):
        """ Extract all data in the storage to save it """
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in range(len(self._storage)):
            data = self._storage[self._save_idx]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        if self._n_step == 0:
            return self._encode_sample(idxes)
        else:
            if self._flg_seq:
                return self._encode_sample_n_step_sequence(idxes)
            else:
                return self._encode_sample_n_step(idxes)

    def save_json(self, filename):
        """
        this is too slow... don't use this unless especially needed

        <Experimental Result>
        - for 1000 time-step in a trajectory, this took 8.8s to save
        """
        o_t, a, r, o_tp1, d = self._get_data_to_save()
        o_t, a, r, o_tp1, d = o_t.tolist(), a.tolist(), r.tolist(), o_tp1.tolist(), d.tolist()
        data = {
            "o": o_t,
            "a": a,
            "r": r,
            "next_o": o_tp1,
            "done": d
        }
        data = json.dumps(data)
        with open(filename, 'w') as fp:
            json.dump(data, fp)

    def save_np(self, _save_id):
        """ this is way quicker than save_json

        <Experimental Result>
        - for 1000 time-step in a trajectory, this took about 0.25s to save
        """
        # o_t, a, r, o_tp1, d = self._get_data_to_save()
        o_t, a, r, o_tp1, d = self._get_all_data()

        if not os.path.isdir(self.traj_dir):
            os.makedirs(self.traj_dir)
        else:
            shutil.rmtree(self.traj_dir)
            os.makedirs(self.traj_dir)

        self._save(o_t, a, r, o_tp1, d)

        # np.save(file=self.traj_dir + "/obs", arr=o_t)
        # np.save(file=self.traj_dir + "/action", arr=a)
        # np.save(file=self.traj_dir + "/reward", arr=r)
        # np.save(file=self.traj_dir + "/obs_tp1", arr=o_tp1)
        # np.save(file=self.traj_dir + "/done", arr=d)

    def _save(self, o_t, a, r, o_tp1, d):
        a = tf.Variable(a)
        o_t = tf.Variable(o_t)
        o_tp1 = tf.Variable(o_tp1)
        r = tf.Variable(r)
        d = tf.Variable(d)

        # register tables to the checkpoint
        _check_point = tf.train.Checkpoint()

        data = {
            "a": a,
            "o_t": o_t,
            "o_tp1": o_tp1,
            "r": r,
            "d": d
        }

        self._summarise_data(data=data)
        _check_point.mapped = data
        tf.train.CheckpointManager(_check_point, self.traj_dir, max_to_keep=1).save()

    def _summarise_data(self, data):
        with open(self.traj_dir+"/var_summary.txt", "w") as f:
            for key, value in data.items():
                txt = self._process_dtype_text(value.dtype)
                f.write("{}\t{}\t{}\n".format(key, txt, value.shape))
            f.close()

    def _process_dtype_text(self, dtype):
        txt = str(dtype)[1:-1]
        txt = txt.replace("'", "")
        txt = txt.split(": ")[1]
        txt = "tf." + txt
        return txt

    def _load_summary(self):
        data = dict()
        with open(self.traj_dir+"/var_summary.txt", "r") as f:
            lines = f.read()
            for line in lines.split("\n")[:-1]:
                var_name, dtype, shape = line.split("\t")
                data[var_name] = tf.Variable(tf.zeros(shape=eval(shape), dtype=eval(dtype)))
        f.close()
        return data

    def load_tf(self):
        # register tables to the checkpoint
        _check_point = tf.train.Checkpoint()
        data = self._load_summary()
        _check_point.mapped = data

        _manager = tf.train.CheckpointManager(_check_point, self.traj_dir, max_to_keep=1)
        _check_point.restore(_manager.latest_checkpoint)
        o_t, a, r, o_tp1, d = data["o_t"].numpy(), data["a"].numpy(), data["r"].numpy(), data["o_tp1"].numpy(), data["d"].numpy()
        for _o_t, _a, _r, _o_tp1, _d in zip(o_t, a, r, o_tp1, d):
            self.add(_o_t, _a, _r, _o_tp1, _d)

    def load_np(self):
        """ load trajectories from traj_dir(root dir for all trajectories stored using save_np) """
        # TODO: according to my experiment(`Test/OtherTest/load_memory_test.py`), this took 37s for loading 10,000 steps
        # don't you think it's toooooo slow???

        if os.path.isdir(self.traj_dir):
            print("Loading: {}".format(self.traj_dir))
            o_t = np.load(file=self.traj_dir + "/obs.npy").tolist()
            a = np.load(file=self.traj_dir + "/action.npy").tolist()
            r = np.load(file=self.traj_dir + "/reward.npy").tolist()
            o_tp1 = np.load(file=self.traj_dir + "/obs_tp1.npy").tolist()
            d = np.load(file=self.traj_dir + "/done.npy").tolist()

            for _o_t, _a, _r, _o_tp1, _d in zip(o_t, a, r, o_tp1, d):
                self.add(_o_t, _a, _r, _o_tp1, _d)

            print("=== Finish: {} steps have been loaded ===".format(len(self._storage)))
        else:
            print("No previous trajectories are found in {}".format(self.traj_dir))

    def refresh(self):
        self._storage = []
        self._next_idx = 0


"""
Hindsight Experience Replay buffer
* the replay buffer here is basically from the openai baselines code
"""
import threading


class HER_replay_buffer:
    def __init__(self, env_params, buffer_size, sample_func):
        self.env_params = env_params
        self.T = env_params['max_timesteps']
        self.size = buffer_size // self.T
        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0
        self.sample_func = sample_func
        # create the buffer to store info
        self.buffers = {'obs': np.empty([self.size, self.T + 1, self.env_params['obs']]),
                        'ag': np.empty([self.size, self.T + 1, self.env_params['goal']]),
                        'g': np.empty([self.size, self.T, self.env_params['goal']]),
                        'actions': np.empty([self.size, self.T, self.env_params['action']]),
                        }
        # thread lock
        self.lock = threading.Lock()

    # store the episode
    def store_episode(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        batch_size = mb_obs.shape[0]
        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)
            # store the informations
            self.buffers['obs'][idxs] = mb_obs
            self.buffers['ag'][idxs] = mb_ag
            self.buffers['g'][idxs] = mb_g
            self.buffers['actions'][idxs] = mb_actions
            self.n_transitions_stored += self.T * batch_size

    # sample the data from the replay buffer
    def sample(self, batch_size):
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size]
        temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]
        temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]
        # sample transitions
        transitions = self.sample_func(temp_buffers, batch_size)
        return transitions

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size + inc <= self.size:
            idx = np.arange(self.current_size, self.current_size + inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx
