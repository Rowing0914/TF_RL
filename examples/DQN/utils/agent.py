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
                 dim_action,
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
        self.dim_action = dim_action
        self.main_model = model(dim_action)
        self.target_model = model(dim_action)
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

    def update(self, states, actions, rewards, next_states, dones):
        states, next_states = self._obs_prc_fn(states), self._obs_prc_fn(next_states)
        states = np.array(states, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        actions = np.array(actions, dtype=np.uint8)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        return self._update(states, actions, rewards, next_states, dones)

    @tf.function
    def _update(self, states, actions, rewards, next_states, dones):
        # get the current global time-step
        self._timestep = tf.compat.v1.train.get_global_step()

        # ===== make sure to fit all process to compute gradients within this Tape context!! =====
        with tf.GradientTape() as tape:
            next_Q = self.target_model(next_states)
            q_values = self.main_model(states)
            Y = rewards + self._gamma * tf.math.reduce_max(next_Q, axis=-1) * (1. - dones)
            Y = tf.stop_gradient(Y)

            # get the q-values which is associated with actually taken actions in a game
            actions_one_hot = tf.one_hot(actions, self.dim_action, 1.0, 0.0)
            chosen_q = tf.math.reduce_sum(tf.math.multiply(actions_one_hot, q_values), axis=-1)
            batch_loss = self._loss_fn(Y, chosen_q, reduction=tf.losses.Reduction.NONE)
            loss = tf.math.reduce_mean(batch_loss)

        # get gradients
        grads = tape.gradient(loss, self.main_model.trainable_weights)

        # clip gradients
        grads = self._grad_clip_fn(grads)

        # apply processed gradients to the network
        self._optimizer.apply_gradients(zip(grads, self.main_model.trainable_weights))

        with tf.name_scope("Agent"):
            tf.compat.v2.summary.scalar("loss", loss, step=self._timestep)
            tf.compat.v2.summary.histogram("next_Q(TargetModel)", next_Q, step=self._timestep)
            tf.compat.v2.summary.histogram("q_values(MainModel)", next_Q, step=self._timestep)
            tf.compat.v2.summary.histogram("Y(target)", Y, step=self._timestep)
            tf.compat.v2.summary.scalar("mean_Y(target)", tf.math.reduce_mean(Y), step=self._timestep)
            tf.compat.v2.summary.scalar("var_Y(target)", tf.math.reduce_variance(Y), step=self._timestep)
            tf.compat.v2.summary.scalar("max_Y(target)", tf.math.reduce_max(Y), step=self._timestep)
            tf.compat.v2.summary.scalar("mean_q_value(TargetModel)", tf.math.reduce_mean(next_Q), step=self._timestep)
            tf.compat.v2.summary.scalar("var_q_value(TargetModel)", tf.math.reduce_variance(next_Q),
                                        step=self._timestep)
            tf.compat.v2.summary.scalar("max_q_value(TargetModel)", tf.math.reduce_max(next_Q), step=self._timestep)
            tf.compat.v2.summary.scalar("mean_q_value(MainModel)", tf.math.reduce_mean(q_values), step=self._timestep)
            tf.compat.v2.summary.scalar("var_q_value(MainModel)", tf.math.reduce_variance(q_values),
                                        step=self._timestep)
            tf.compat.v2.summary.scalar("max_q_value(MainModel)", tf.math.reduce_max(q_values), step=self._timestep)

        return loss, batch_loss
