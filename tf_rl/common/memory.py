"""
Design reference
https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
"""

import tensorflow as tf
import numpy as np
import shutil
import random
import os

from tf_rl.common.segment_tree import SumSegmentTree, MinSegmentTree

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
                 traj_dir="./tmp",
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
            self.load()

    def __len__(self):
        return len(self._storage)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # just in case, if something wrong happens, we'd save the data
        self.save()

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

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

            # o_seq, a_seq, r_seq, no_seq, d_seq = data_n  <= this doesn't work....

            # TODO: this is not efficient... because every index, we go through n-step
            # so that consider using separate bucket for each component(o, a, r, no, d)
            # then we can just specify the indices of them instead of having for-loop to operate on them.
            o_seq, a_seq, r_seq, no_seq, d_seq = [], [], [], [], []
            for _data in data_n:
                _o, _a, _r, _no, _d = _data
                o_seq.append(np.array(_o, copy=False))
                a_seq.append(_a)
                r_seq.append(_r)
                no_seq.append(np.array(_no, copy=False))
                d_seq.append(_d)

            o_seq, a_seq, r_seq, no_seq, d_seq = self._check_episode_end(o_seq, a_seq, r_seq, no_seq, d_seq)

            # Store a data at each time-sequence in the resulting array
            obs_t.append(np.array(o_seq, copy=False))
            action.append(a_seq)
            reward.append(r_seq)
            obs_tp1.append(np.array(no_seq, copy=False))
            done.append(d_seq)
        return np.array(obs_t), np.array(action), np.array(reward), np.array(obs_tp1), np.array(done)

    def _check_episode_end(self, o_seq, a_seq, r_seq, no_seq, d_seq):
        """ validate if the extracted part of the memory is semantically correct. """
        _id = np.argmax(np.asarray(d_seq).astype(np.float32))
        if _id == 0:
            return o_seq, a_seq, no_seq, r_seq, d_seq
        else:
            o_seq = self._zero_padding(_id, o_seq)
            a_seq = self._zero_padding(_id, a_seq)
            no_seq = self._zero_padding(_id, no_seq)
            r_seq = self._zero_padding(_id, r_seq)
            d_seq = self._zero_padding(_id, d_seq)
            return o_seq, a_seq, no_seq, r_seq, d_seq

    def _zero_padding(self, index, data):
        """ Pad the states after termination by 0s """
        _sample = np.asarray(data[0])
        _dtype, _shape = _sample.dtype, _sample.shape
        before_terminate = data[:index]
        after_terminate = np.zeros(shape=(self._n_step-index,)+_shape, dtype=_dtype).tolist()
        return np.asarray(before_terminate + after_terminate)

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

    def _get_all_data(self):
        """ Extract all data in the storage to save it """
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in range(len(self._storage)):
            data = self._storage[i]
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

    def save(self):
        """ save method """
        o_t, a, r, o_tp1, d = self._get_all_data()

        if not os.path.isdir(self.traj_dir):
            os.makedirs(self.traj_dir)
        else:
            shutil.rmtree(self.traj_dir)
            os.makedirs(self.traj_dir)

        self._save(o_t, a, r, o_tp1, d)

    def _save(self, o_t, a, r, o_tp1, d):
        # a = tf.Variable(a)
        # o_t = tf.Variable(o_t)
        # o_tp1 = tf.Variable(o_tp1)
        # r = tf.Variable(r)
        # d = tf.Variable(d)

        # register tables to the checkpoint
        _check_point = tf.train.Checkpoint()

        data = {
            "a": tf.Variable(a),
            "o_t": tf.Variable(o_t),
            "o_tp1": tf.Variable(o_tp1),
            "r": tf.Variable(r),
            "d": tf.Variable(d)
        }

        self._summarise_data(data=data)
        _check_point.mapped = data
        del data  # garbage collection
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

    def load(self):
        # register tables to the checkpoint
        _check_point = tf.train.Checkpoint()
        data = self._load_summary()
        _check_point.mapped = data

        _manager = tf.train.CheckpointManager(_check_point, self.traj_dir, max_to_keep=1)
        _check_point.restore(_manager.latest_checkpoint)
        o_t, a, r, o_tp1, d = data["o_t"].numpy(), data["a"].numpy(), data["r"].numpy(), data["o_tp1"].numpy(), data["d"].numpy()
        del data  # garbage collection
        for _o_t, _a, _r, _o_tp1, _d in zip(o_t, a, r, o_tp1, d):
            self.add(_o_t, _a, _r, _o_tp1, _d)

    def refresh(self):
        self._storage = []
        self._next_idx = 0


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha, n_step=0, gamma=0.99):
        """Create Prioritized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(size, n_step, gamma)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, len(self._storage) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        """Sample a batch of experiences.
        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
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
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        if self._n_step == 0:
            encoded_sample = self._encode_sample(idxes)
        else:
            encoded_sample = self._encode_sample_n_step(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)



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
