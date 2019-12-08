"""
Design reference
https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
"""

import numpy as np
import random
import json

class ReplayBuffer(object):
    def __init__(self, size, n_step=0, gamma=0.99, flg_seq=True):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._n_step = n_step
        self._gamma = gamma
        self._next_idx = 0
        self._flg_seq = flg_seq

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

            # TODO: this is not efficient... because every index, we go through n-step
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

    def save(self, dir, save_amount=10000):
        """
        maybe don't use this!!! memory consumption is disastrous....

        :param dir:
        :param save_amount:
        :return:
        """
        obses_t, actions, rewards, obses_tp1, dones = self.sample(batch_size=save_amount)
        obses_t, actions, rewards, obses_tp1, dones = obses_t.tolist(), actions.tolist(), rewards.tolist(), obses_tp1.tolist(), dones.tolist()
        data = {
            "index": save_amount,
            "o": obses_t,
            "a": actions,
            "r": rewards,
            "next_o": obses_tp1,
            "done": dones
        }
        data = json.dumps(data)
        json.dump(data, dir)

    def refresh(self):
        self._storage = []
        self._next_idx = 0