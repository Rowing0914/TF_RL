import numpy as np
import tensorflow as tf
from copy import deepcopy
from tf_rl.common.utils import create_checkpoint
from tf_rl.agents.core import Agent


class SAC(Agent):
    def __init__(self, actor, critic, num_action, params):
        self.params = params
        self.num_action = num_action
        self.eval_flg = False
        self.index_timestep = 0
        self.actor = actor(num_action)
        self.critic = critic(1)
        # self.target_actor = deepcopy(self.actor)
        self.target_critic = deepcopy(self.critic)
        self.actor_optimizer = tf.train.AdamOptimizer(learning_rate=3e-4)  # used as in paper
        self.critic_optimizer = tf.train.AdamOptimizer(learning_rate=3e-4)  # used as in paper
        self.actor_manager = create_checkpoint(model=self.actor,
                                               optimizer=self.actor_optimizer,
                                               model_dir=params.actor_model_dir)
        self.critic_manager = create_checkpoint(model=self.critic,
                                                optimizer=self.critic_optimizer,
                                                model_dir=params.critic_model_dir)

    def predict(self, state):
        state = np.expand_dims(state, axis=0).astype(np.float32)
        action, _, _ = self._select_action(tf.constant(state))
        return action.numpy()[0]

    def eval_predict(self, state):
        """
        As mentioned in the topic of `policy evaluation` at sec5.2(`ablation study`) in the paper,
        for evaluation phase, using a deterministic action(choosing the mean of the policy dist) works better than
        stochastic one(Gaussian Policy).
        """
        state = np.expand_dims(state, axis=0).astype(np.float32)
        _, _, action = self._select_action(tf.constant(state))
        return action.numpy()[0]

    @tf.contrib.eager.defun(autograph=False)
    def _select_action(self, state):
        return self.actor(state)

    @tf.contrib.eager.defun(autograph=False)
    def _inner_update(self, states, actions, rewards, next_states, dones):
        self.index_timestep = tf.train.get_global_step()
        # Update Critic
        with tf.GradientTape() as tape:
            # critic takes as input states, actions so that we combine them before passing them
            next_action, next_state_log_pi, _ = self.actor(next_states)
            next_Q1, next_Q2 = self.target_critic(next_states, next_action)
            min_next_Q_target = tf.math.minimum(next_Q1, next_Q2) - self.params.alpha * next_state_log_pi
            q1, q2 = self.critic(states, actions)

            # compute the target discounted Q(s', a')
            Y = rewards + self.params.gamma * tf.reshape(min_next_Q_target, [-1]) * (1. - dones)
            Y = tf.stop_gradient(Y)

            # Compute critic loss
            critic_loss_q1 = tf.losses.mean_squared_error(Y, tf.reshape(q1, [-1]))
            critic_loss_q2 = tf.losses.mean_squared_error(Y, tf.reshape(q2, [-1]))

        critic_grads = tape.gradient([critic_loss_q1, critic_loss_q2], self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        # Update Actor
        with tf.GradientTape() as tape:
            action, log_pi, _ = self.actor(states)
            q1, q2 = self.critic(states, action)
            actor_loss = tf.math.reduce_mean((self.params.alpha * log_pi) - tf.math.minimum(q1, q2))

        # get gradients
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)

        # apply processed gradients to the network
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        return tf.math.reduce_sum(critic_loss_q1 + critic_loss_q2 + actor_loss)


class SAC_debug(Agent):
    def __init__(self, actor, critic, num_action, params):
        self.params = params
        self.num_action = num_action
        self.eval_flg = False
        self.index_timestep = 0
        self.actor = actor(num_action)
        self.critic = critic(1)
        self.target_actor = deepcopy(self.actor)
        self.target_critic = deepcopy(self.critic)
        self.actor_optimizer = tf.train.AdamOptimizer(learning_rate=3e-4)  # used as in paper
        self.critic_optimizer = tf.train.AdamOptimizer(learning_rate=3e-4)  # used as in paper
        self.actor_manager = create_checkpoint(model=self.actor,
                                               optimizer=self.actor_optimizer,
                                               model_dir=params.actor_model_dir)
        self.critic_manager = create_checkpoint(model=self.critic,
                                                optimizer=self.critic_optimizer,
                                                model_dir=params.critic_model_dir)

    def predict(self, state):
        state = np.expand_dims(state, axis=0).astype(np.float32)
        action, _, _ = self._select_action(tf.constant(state))
        return action.numpy()[0]

    def eval_predict(self, state):
        """
        As mentioned in the topic of `policy evaluation` at sec5.2(`ablation study`) in the paper,
        for evaluation phase, using a deterministic action(choosing the mean of the policy dist) works better than
        stochastic one(Gaussian Policy).
        """
        state = np.expand_dims(state, axis=0).astype(np.float32)
        _, _, action = self._select_action(tf.constant(state))
        return action.numpy()[0]

    @tf.contrib.eager.defun(autograph=False)
    def _select_action(self, state):
        return self.actor(state)

    @tf.contrib.eager.defun(autograph=False)
    def _inner_update(self, states, actions, rewards, next_states, dones):
        self.index_timestep = tf.train.get_global_step()
        # Update Critic
        with tf.GradientTape() as tape:
            # critic takes as input states, actions so that we combine them before passing them
            next_action, next_state_log_pi, _ = self.actor(next_states)
            next_Q1, next_Q2 = self.target_critic(next_states, next_action)
            next_Q = tf.math.minimum(next_Q1, next_Q2) - self.params.alpha * next_state_log_pi
            q1, q2 = self.critic(states, actions)

            # compute the target discounted Q(s', a')
            Y = rewards + self.params.gamma * tf.reshape(next_Q, [-1]) * (1. - dones)
            Y = tf.stop_gradient(Y)

            # Compute critic loss
            critic_loss_q1 = tf.losses.mean_squared_error(Y, tf.reshape(q1, [-1]))
            critic_loss_q2 = tf.losses.mean_squared_error(Y, tf.reshape(q2, [-1]))

        critic_grads = tape.gradient([critic_loss_q1, critic_loss_q2], self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        # Update Actor
        with tf.GradientTape() as tape:
            action, log_pi, _ = self.actor(states)
            q1, q2 = self.critic(states, action)
            actor_loss = tf.math.reduce_mean((self.params.alpha * log_pi) - tf.math.minimum(q1, q2))

        # get gradients
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)

        # apply processed gradients to the network
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        critic_grads = tf.math.reduce_mean([tf.math.reduce_sum(grad) for grad in critic_grads])
        actor_grads = tf.math.reduce_mean([tf.math.reduce_sum(grad) for grad in actor_grads])

        tf.contrib.summary.histogram("Y", Y, step=self.index_timestep)
        # tf.contrib.summary.histogram("next_action", next_action, step=self.index_timestep)
        # tf.contrib.summary.histogram("next_state_log_pi", next_state_log_pi, step=self.index_timestep)
        # tf.contrib.summary.histogram("next_Q1", next_Q1, step=self.index_timestep)
        # tf.contrib.summary.histogram("next_Q2", next_Q2, step=self.index_timestep)
        tf.contrib.summary.scalar("critic_loss_q1", critic_loss_q1, step=self.index_timestep)
        tf.contrib.summary.scalar("critic_loss_q2", critic_loss_q2, step=self.index_timestep)
        tf.contrib.summary.scalar("actor_loss", actor_loss, step=self.index_timestep)

        tf.contrib.summary.scalar("critic_grad", critic_grads, step=self.index_timestep)
        tf.contrib.summary.scalar("actor_grad", actor_grads, step=self.index_timestep)

        # tf.contrib.summary.scalar("mean_next_Q", tf.math.reduce_mean(next_Q), step=self.index_timestep)
        # tf.contrib.summary.scalar("max_next_Q", tf.math.reduce_max(next_Q), step=self.index_timestep)
        tf.contrib.summary.scalar("mean_q1", tf.math.reduce_mean(q1), step=self.index_timestep)
        tf.contrib.summary.scalar("mean_q2", tf.math.reduce_mean(q2), step=self.index_timestep)
        tf.contrib.summary.scalar("max_q1", tf.math.reduce_max(q1), step=self.index_timestep)
        tf.contrib.summary.scalar("max_q2", tf.math.reduce_max(q2), step=self.index_timestep)

        # print(q1.numpy(), q2.numpy(), critic_loss_q1.numpy(), critic_loss_q2.numpy(), actor_loss.numpy())
        return tf.math.reduce_sum(critic_loss_q1 + critic_loss_q2 + actor_loss)
