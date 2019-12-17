import numpy as np
import tensorflow as tf
from copy import deepcopy
from tf_rl.common.utils import create_checkpoint


class DDPG(object):
    def __init__(self,
                 actor,
                 critic,
                 num_action,
                 random_process,
                 gamma,
                 L2_reg,
                 actor_model_dir,
                 critic_model_dir):
        self._num_action = num_action
        self._gamma = gamma
        self._L2_reg = L2_reg

        self.global_ts = tf.compat.v1.train.get_or_create_global_step()
        self.eval_flg = False
        self.actor = actor(num_action)
        self.critic = critic(1)
        # self.target_actor = deepcopy(self.actor)
        # self.target_critic = deepcopy(self.critic)
        self.target_actor = actor(num_action)
        self.target_critic = critic(1)
        self.actor_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4)
        self.critic_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
        self.random_process = random_process
        self.actor_manager = create_checkpoint(model=self.actor,
                                               optimizer=self.actor_optimizer,
                                               model_dir=actor_model_dir)
        self.critic_manager = create_checkpoint(model=self.critic,
                                                optimizer=self.critic_optimizer,
                                                model_dir=critic_model_dir)

    def select_action(self, state):
        state = np.expand_dims(state, axis=0).astype(np.float32)
        action = self._select_action(tf.constant(state))
        return action.numpy()[0] + self.random_process.sample()

    def select_action_eval(self, state):
        """ Deterministic behaviour """
        state = np.expand_dims(state, axis=0).astype(np.float32)
        action = self._select_action(tf.constant(state))
        return action.numpy()[0]

    @tf.function
    def _select_action(self, state):
        return self.actor(state)

    def update(self, states, actions, rewards, next_states, dones):
        """
        Update methods for Actor and Critic
        please refer to https://arxiv.org/pdf/1509.02971.pdf about the details

        """
        states = np.array(states, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        actions = np.array(actions, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        return self._inner_update(states, actions, rewards, next_states, dones)

    @tf.function
    def _inner_update(self, states, actions, rewards, next_states, dones):
        self.global_ts = tf.compat.v1.train.get_global_step()
        # Update Critic
        with tf.GradientTape() as tape:
            # critic takes as input states, actions so that we combine them before passing them
            next_Q = self.target_critic(next_states, self.target_actor(next_states))
            q_values = self.critic(states, actions)

            # compute the target discounted Q(s', a')
            Y = rewards + self._gamma * tf.reshape(next_Q, [-1]) * (1. - dones)
            Y = tf.stop_gradient(Y)

            # Compute critic loss(MSE or huber_loss) + L2 loss
            critic_loss = tf.compat.v1.losses.mean_squared_error(Y, tf.reshape(q_values, [-1])) + tf.add_n(
                self.critic.losses) * self._L2_reg

        # get gradients
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)

        # apply processed gradients to the network
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        # Update Actor
        with tf.GradientTape() as tape:
            actor_loss = -tf.math.reduce_mean(self.critic(states, self.actor(states)))

        # get gradients
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)

        # apply processed gradients to the network
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        return tf.math.reduce_sum(critic_loss + actor_loss)
