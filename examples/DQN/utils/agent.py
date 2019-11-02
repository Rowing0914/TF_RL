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
        action = self.policy.select_action(q_value_fn=self._select_action, state=state, epsilon=epsilon)
        return action

    @tf.function
    def _select_action(self, state):
        return self.main_model(state)

    @tf.function
    def update(self, states, actions, rewards, next_states, dones):
        states = tf.cast(states, dtype=tf.float32)  # batch_size x w x h x c
        next_states = tf.cast(next_states, dtype=tf.float32)  # batch_size x w x h x c
        actions = tf.cast(actions, dtype=tf.int32)  # (batch_size,)
        rewards = tf.cast(rewards, dtype=tf.float32)  # (batch_size,)
        dones = tf.cast(dones, dtype=tf.float32)  # (batch_size,)
        states, next_states = self._obs_prc_fn(states), self._obs_prc_fn(next_states)
        return self._update(states, actions, rewards, next_states, dones)

    def _update(self, states, actions, rewards, next_states, dones):
        # ===== make sure to fit all process to compute gradients within this Tape context!! =====
        with tf.GradientTape() as tape:
            q_tp1 = self.target_model(next_states)  # batch_size x num_action
            q_t = self.main_model(states)  # batch_size x num_action
            Y = rewards + self._gamma * tf.math.reduce_max(q_tp1, axis=-1) * (1. - dones)  # (batch_size,)
            Y = tf.stop_gradient(Y)

            # get the q-values which is associated with actually taken actions in a game
            actions_one_hot = tf.one_hot(actions, self.num_action, 1.0, 0.0)  # batch_size x num_action
            chosen_q = tf.math.reduce_sum(actions_one_hot * q_t, axis=-1)  # (batch_size,)
            batch_loss = self._loss_fn(Y, chosen_q, reduction=tf.compat.v1.losses.Reduction.NONE)  # (batch_size,)
            loss = tf.math.reduce_mean(batch_loss)

        # get gradients
        grads = tape.gradient(loss, self.main_model.trainable_weights)

        # clip gradients
        grads = self._grad_clip_fn(grads)

        # apply processed gradients to the network
        self._optimizer.apply_gradients(zip(grads, self.main_model.trainable_weights))

        # get the current global time-step
        ts = self._timestep = tf.compat.v1.train.get_global_step()
        tf.compat.v2.summary.scalar("agent/loss", loss, step=ts)
        tf.compat.v2.summary.scalar("agent/mean q_tp1 - q_t", tf.math.reduce_mean(q_tp1 - q_t), step=ts)
        tf.compat.v2.summary.scalar("agent/mean Y", tf.math.reduce_mean(Y), step=ts)
        tf.compat.v2.summary.scalar("agent/mean q_tp1", tf.math.reduce_mean(q_tp1), step=ts)
        tf.compat.v2.summary.scalar("agent/mean q_t", tf.math.reduce_mean(q_t), step=ts)

        return loss, batch_loss
