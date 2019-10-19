"""
Design reference
https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


class Table:
    """ Base table class to store data in tf.Variable """

    def __init__(self, capacity, shape, dtype):
        self._capacity = capacity
        self._shape = (capacity,) + shape
        self._dtype = dtype
        self._storage = tf.Variable(tf.zeros(self._shape, dtype=dtype), dtype=dtype)

    def write(self, row, value):
        tf.compat.v1.scatter_update(self._storage, row, value)

    def read_all(self):
        return self._storage.read_value()

    def read_row(self, row):
        return tf.gather(self._storage, row)

    def read_seq(self, _start, _end):
        return tf.gather(self._storage, tf.range(start=_start, limit=_end))

    def refresh(self):
        self._storage = tf.Variable(tf.zeros(self._shape, dtype=self._dtype), dtype=self._dtype)


class Buffer:
    def __init__(self,
                 capacity,
                 n_step,
                 act_shape,
                 obs_shape,
                 act_dtype=tf.float32,
                 obs_dtype=tf.float32,
                 reward_shape=(),
                 reward_dtype=tf.float32,
                 done_shape=(),
                 done_dtype=tf.int32,
                 checkpoint_dir="./tmp/"):
        self._current_idx = tf.Variable(0)
        self._len_idx = tf.Variable(0)
        self._capacity = capacity
        self._n_step = n_step
        self._act_shape = act_shape
        self._act_dtype = act_dtype
        self._obs_shape = obs_shape
        self._obs_dtype = obs_dtype
        self._reward_shape = reward_shape
        self._reward_dtype = reward_dtype
        self._done_shape = done_shape
        self._done_dtype = done_dtype

        # create tables
        self._action_table = Table(capacity=capacity, shape=act_shape, dtype=act_dtype)
        self._obs_table = Table(capacity=capacity, shape=obs_shape, dtype=obs_dtype)
        self._next_obs_table = Table(capacity=capacity, shape=obs_shape, dtype=obs_dtype)
        self._reward_table = Table(capacity=capacity, shape=reward_shape, dtype=reward_dtype)
        self._done_table = Table(capacity=capacity, shape=done_shape, dtype=done_dtype)

        # register tables to the checkpoint
        self._check_point = tf.train.Checkpoint()

        self._check_point.mapped = {
            "current_idx": self._current_idx,
            "len_idx": self._len_idx,
            "action_table": self._action_table._storage,
            "obs_table": self._obs_table._storage,
            "next_obs_table": self._next_obs_table._storage,
            "reward_table": self._reward_table._storage,
            "done_table": self._done_table._storage
        }

        self.checkpoint_dir = checkpoint_dir
        self.manager = tf.train.CheckpointManager(self._check_point, checkpoint_dir, max_to_keep=1)
        self._try_restore()

    def __len__(self):
        return self._len_idx.read_value()

    def read_all(self):
        data = dict()
        data.update({"actions": self._action_table.read_all().numpy()})
        data.update({"obs": self._obs_table.read_all().numpy()})
        data.update({"next_obs": self._next_obs_table.read_all().numpy()})
        data.update({"rewards": self._reward_table.read_all().numpy()})
        data.update({"dones": self._done_table.read_all().numpy()})
        return data

    def refresh(self):
        self._action_table.refresh()
        self._obs_table.refresh()
        self._next_obs_table.refresh()
        self._reward_table.refresh()
        self._done_table.refresh()

    def _try_restore(self):
        self._check_point.restore(self.manager.latest_checkpoint)

    def save(self):
        return self.manager.save()

    def add(self, obs_t, action, reward, obs_tp1, done):
        # 1. validate the position of the cursor
        self._current_idx.assign((self._current_idx + 1) % self._capacity)

        # 2. insert the data to the position
        self._write(self._current_idx, obs_t, action, reward, obs_tp1, done)

        # 3. maintain the index representing the length of the buffer
        if self._len_idx < self._capacity:
            self._len_idx.assign_add(1)

    def _validate_dtype(self, value, dtype):
        return tf.cast(value, dtype=dtype)

    def _write(self, row, obs, next_obs, action, reward, done):
        obs = self._validate_dtype(value=obs, dtype=self._obs_dtype)
        next_obs = self._validate_dtype(value=next_obs, dtype=self._obs_dtype)
        action = self._validate_dtype(value=action, dtype=self._act_dtype)
        reward = self._validate_dtype(value=reward, dtype=self._reward_dtype)
        done = self._validate_dtype(value=done, dtype=self._done_dtype)

        self._obs_table.write(row, obs)
        self._next_obs_table.write(row, next_obs)
        self._action_table.write(row, action)
        self._reward_table.write(row, reward)
        self._done_table.write(row, done)

    def _read_row(self, row):
        obs = self._obs_table.read_row(row)
        next_obs = self._next_obs_table.read_row(row)
        action = self._action_table.read_row(row)
        reward = self._reward_table.read_row(row)
        done = self._done_table.read_row(row)
        return obs, action, next_obs, reward, done

    def _read_seq(self, _start, _end):
        obs_seq = self._obs_table.read_seq(_start, _end)
        next_obs_seq = self._next_obs_table.read_seq(_start, _end)
        action_seq = self._action_table.read_seq(_start, _end)
        reward_seq = self._reward_table.read_seq(_start, _end)
        done_seq = self._done_table.read_seq(_start, _end)
        return obs_seq, action_seq, next_obs_seq, reward_seq, done_seq

    @tf.function
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
        # generate random numbers to collect samples from the storages
        idxes = tf.py_function(func=lambda: np.random.randint(low=0, high=self._len_idx-1, size=batch_size),
                               inp=[],
                               Tout=tf.int32)
        if self._n_step == 0:
            return self._encode_sample(idxes, batch_size)
        else:
            return self._encode_sample_n_step_sequence(idxes, batch_size)

    def _encode_sample(self, idxes, batch_size):
        """ One step sampling method """
        obses_t = tf.TensorArray(dtype=self._obs_dtype, size=batch_size)
        actions = tf.TensorArray(dtype=self._act_dtype, size=batch_size)
        rewards = tf.TensorArray(dtype=self._reward_dtype, size=batch_size)
        obses_tp1 = tf.TensorArray(dtype=self._obs_dtype, size=batch_size)
        dones = tf.TensorArray(dtype=self._done_dtype, size=batch_size)

        for id in range(batch_size):
            obs_t, action, obs_tp1, reward, done = self._read_row(row=tf.gather(idxes, id))
            obses_t = obses_t.write(id, obs_t)
            actions = actions.write(id, action)
            rewards = rewards.write(id, reward)
            obses_tp1 = obses_tp1.write(id, obs_tp1)
            dones = dones.write(id, done)
        return obses_t.stack(), actions.stack(), rewards.stack(), obses_tp1.stack(), dones.stack()

    def _encode_sample_n_step_sequence(self, idxes, batch_size):
        """
        n-consecutive time-step sampling method
        Return:
            obs, act, rew, next_obs, done FROM t to t+n
        """
        obses_t = tf.TensorArray(dtype=self._obs_dtype, size=batch_size)
        actions = tf.TensorArray(dtype=self._act_dtype, size=batch_size)
        rewards = tf.TensorArray(dtype=self._reward_dtype, size=batch_size)
        obses_tp1 = tf.TensorArray(dtype=self._obs_dtype, size=batch_size)
        dones = tf.TensorArray(dtype=self._done_dtype, size=batch_size)

        # === Sampling method ===
        for id in range(batch_size):
            index = tf.gather(idxes, id)
            if index + self._n_step >= self._len_idx:  # avoid the index out of range error!!
                first_half = self._len_idx - 1 - index
                second_half = self._n_step - first_half
                o_seq1, a_seq1, no_seq1, r_seq1, d_seq1 = self._read_seq(_start=index, _end=index+first_half)
                o_seq2, a_seq2, no_seq2, r_seq2, d_seq2 = self._read_seq(_start=0, _end=second_half)
                o_seq = tf.concat([o_seq1, o_seq2], axis=0)
                a_seq = tf.concat([a_seq1, a_seq2], axis=0)
                no_seq = tf.concat([no_seq1, no_seq2], axis=0)
                r_seq = tf.concat([r_seq1, r_seq2], axis=0)
                d_seq = tf.concat([d_seq1, d_seq2], axis=0)
            else:
                o_seq, a_seq, no_seq, r_seq, d_seq = self._read_seq(_start=index, _end=index + self._n_step)

            # check if the specified range contains the terminate state
            o_seq, a_seq, no_seq, r_seq, d_seq = self._check_episode_end(o_seq, a_seq, no_seq, r_seq, d_seq)

            obses_t = obses_t.write(id, o_seq)
            actions = actions.write(id, a_seq)
            rewards = rewards.write(id, r_seq)
            obses_tp1 = obses_tp1.write(id, no_seq)
            dones = dones.write(id, d_seq)
        return obses_t.stack(), actions.stack(), rewards.stack(), obses_tp1.stack(), dones.stack()

    def _check_episode_end(self, o_seq, a_seq, no_seq, r_seq, d_seq):
        """ validate if the extracted part of the memory from _start to _end is semantically correct. """
        _id = tf.argmax(d_seq)
        if _id == 0:
            return o_seq, a_seq, no_seq, r_seq, d_seq
        else:
            o_seq = self._zero_padding(_id, o_seq, self._obs_dtype, self._obs_shape)
            a_seq = self._zero_padding(_id, a_seq, self._act_dtype, self._act_shape)
            no_seq = self._zero_padding(_id, no_seq, self._obs_dtype, self._obs_shape)
            r_seq = self._zero_padding(_id, r_seq, self._reward_dtype, self._reward_shape)
            d_seq = self._zero_padding(_id, d_seq, self._done_dtype, self._done_shape)
            return o_seq, a_seq, no_seq, r_seq, d_seq

    def _zero_padding(self, index, data, dtype, shape):
        """ Pad the states after termination by 0s """
        before_terminate = tf.gather(data, tf.range(start=0, limit=index))
        after_terminate = tf.zeros(shape=(self._n_step-index,)+shape, dtype=dtype)
        return tf.concat([before_terminate, after_terminate], axis=0)