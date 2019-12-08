import numpy as np
import tensorflow as tf
from copy import deepcopy
from tf_rl.common.utils import create_checkpoint


class DDPG:
    def __init__(self, ggnn, critic, node_info, num_action, params):
        self.params = params
        self.num_action = num_action
        self.eval_flg = False
        self.index_timestep = 0
        self.actor = ggnn(state_dim=params.num_node_features,
                          node_info=node_info,
                          rec_hidden_unit=params.rec_hidden_unit,
                          rec_output_unit=params.rec_output_unit,
                          recurrent_step=params.recurrent_step)
        self.critic = critic(1)
        self.target_actor = deepcopy(self.actor)
        self.target_critic = deepcopy(self.critic)
        self.actor_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4)
        self.critic_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
        self.actor_manager = create_checkpoint(model=self.actor,
                                               optimizer=self.actor_optimizer,
                                               model_dir=params.actor_model_dir)
        self.critic_manager = create_checkpoint(model=self.critic,
                                                optimizer=self.critic_optimizer,
                                                model_dir=params.critic_model_dir)

    def predict(self, state):
        action = self._select_action(tf.constant(state[np.newaxis, ...].astype(np.float32)))
        return action.numpy().flatten()

    def eval_predict(self, state):
        action = self._select_action(tf.constant(state[np.newaxis, ...].astype(np.float32)))
        return action.numpy().flatten()

    @tf.contrib.eager.defun(autograph=False)
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

    @tf.contrib.eager.defun(autograph=False)
    def _inner_update(self, states, actions, rewards, next_states, dones):
        self.index_timestep = tf.compat.v1.train.get_global_step()
        # Update Critic
        with tf.GradientTape() as tape:
            # critic takes as input states, actions so that we combine them before passing them
            next_Q = self.target_critic(next_states, self.target_actor(tf.reshape(next_states, [32, 1, 97]))[0])
            q_values = self.critic(states, actions)

            # compute the target discounted Q(s', a')
            Y = rewards + self.params.gamma * tf.compat.v1.squeeze(next_Q, [-1]) * (1. - dones)
            Y = tf.stop_gradient(Y)

            # Compute critic loss(MSE or huber_loss) + L2 loss
            critic_loss = tf.compat.v1.losses.mean_squared_error(Y, tf.compat.v1.squeeze(q_values, [-1])) + tf.add_n(
                self.critic.losses) * self.params.L2_reg

        # get gradients
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)

        # apply processed gradients to the network
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        # Update Actor
        with tf.GradientTape() as tape:
            actor_loss = -tf.math.reduce_mean(self.critic(states, self.actor(tf.reshape(states, [32, 1, 97]))[0]))

        # get gradients
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)

        # apply processed gradients to the network
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        return np.sum(critic_loss + actor_loss)
