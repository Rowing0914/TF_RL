import numpy as np
import tensorflow as tf
from tf_rl.common.utils import create_checkpoint


class DQN(object):

    def __init__(self,
                 model,
                 policy,
                 optimizer,
                 loss_fn,
                 grad_clip_fn,
                 obs_prc_fn,
                 num_action,
                 model_dir,
                 gamma):
        self._gamma = gamma
        self._grad_clip_fn = grad_clip_fn
        self._loss_fn = loss_fn
        self._timestep = 0
        self._optimizer = optimizer
        self._obs_prc_fn = obs_prc_fn

        # === Supposed to access from outside ===
        self.policy = policy
        self.eval_flg = False
        self.num_action = num_action
        self.main_model = model(num_action)
        self.target_model = model(num_action)
        self.manager = create_checkpoint(model=self.main_model,
                                         optimizer=self._optimizer,
                                         model_dir=model_dir)

    def select_action(self, state):
        state = np.expand_dims(self._obs_prc_fn(state), axis=0).astype(np.float32)
        action = self.policy.select_action(q_value_fn=self._select_action, state=state)
        return action

    def select_action_eval(self, state, epsilon):
        state = np.expand_dims(self._obs_prc_fn(state), axis=0).astype(np.float32)
        action = self.policy.select_action(q_values=self._select_action, state=state, epsilon=epsilon)
        return action

    @tf.function
    def _select_action(self, state):
        return self.main_model(state)

    def update(self, states, actions, rewards, next_states, dones):
        states = np.array(states, dtype=np.float32)  # batch_size x w x h x c
        next_states = np.array(next_states, dtype=np.float32)  # batch_size x w x h x c
        actions = np.array(actions, dtype=np.int32)  # (batch_size,)
        rewards = np.array(rewards, dtype=np.float32)  # (batch_size,)
        dones = np.array(dones, dtype=np.float32)  # (batch_size,)
        states, next_states = self._obs_prc_fn(states), self._obs_prc_fn(next_states)
        return self._update(states, actions, rewards, next_states, dones)

    @tf.function
    def _update(self, states, actions, rewards, next_states, dones):
        # ===== make sure to fit all process to compute gradients within this Tape context!! =====
        with tf.GradientTape() as tape:
            q_tp1 = self.target_model(next_states)  # batch_size x num_action
            q_t = self.main_model(states)  # batch_size x num_action
            td_target = rewards + self._gamma * tf.math.reduce_max(q_tp1, axis=-1) * (1. - dones)  # (batch_size,)
            td_target = tf.stop_gradient(td_target)

            # get the q-values which is associated with actually taken actions in a game
            idx = tf.concat([tf.expand_dims(tf.range(0, actions.shape[0]), 1), tf.expand_dims(actions, 1)], axis=-1)
            chosen_q = tf.gather_nd(q_t, idx)  # (batch_size,)
            td_error = self._loss_fn(td_target, chosen_q)  # scalar

        # get gradients
        grads = tape.gradient(td_error, self.main_model.trainable_variables)

        # clip gradients
        grads = self._grad_clip_fn(grads)

        # apply processed gradients to the network
        self._optimizer.apply_gradients(zip(grads, self.main_model.trainable_variables))

        # get the current global time-step
        ts = self._timestep = tf.compat.v1.train.get_global_step()
        tf.compat.v2.summary.scalar("agent/loss_td_error", td_error, step=ts)
        tf.compat.v2.summary.scalar("agent/mean_diff_q_tp1_q_t", tf.math.reduce_mean(q_tp1 - q_t), step=ts)
        tf.compat.v2.summary.scalar("agent/mean_td_target", tf.math.reduce_mean(td_target), step=ts)
        tf.compat.v2.summary.scalar("agent/mean_q_tp1", tf.math.reduce_mean(q_tp1), step=ts)
        tf.compat.v2.summary.scalar("agent/mean_q_t", tf.math.reduce_mean(q_t), step=ts)
        tf.compat.v2.summary.scalar("train/Eps", self.policy._epsilon_fn(), step=ts)
