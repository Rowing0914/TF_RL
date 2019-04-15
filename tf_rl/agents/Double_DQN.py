import numpy as np
import tensorflow as tf
from tf_rl.common.utils import huber_loss, ClipIfNotNone

class Double_DQN:
    """
    Double_DQN
    """
    def __init__(self, main_model, target_model, num_action, params):
        self.num_action = num_action
        self.params = params
        self.main_model = main_model(num_action)
        self.target_model = target_model(num_action)
        self.optimizer = tf.train.AdamOptimizer()
        # self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.index_episode = 0

    def predict(self, state):
        return self.main_model(tf.convert_to_tensor(state[None,:], dtype=tf.float32)).numpy()[0]

    def update(self, states, actions, rewards, next_states, dones):
        with tf.GradientTape() as tape:
            # make sure to fit all process to compute gradients within this Tape context!!

            # this is where Double DQN comes in!!
            # calculate target: R + gamma * max_a Q(s', max_a Q(s', a'; main_model); target_model)
            next_Q_main = self.main_model(tf.convert_to_tensor(next_states, dtype=tf.float32))
            next_Q = self.target_model(tf.convert_to_tensor(next_states, dtype=tf.float32))
            idx_flattened = tf.range(0, tf.shape(next_Q)[0]) * tf.shape(next_Q)[1] + np.argmax(next_Q_main, axis=-1)

            # passing [-1] to tf.reshape means flatten the array
            # using tf.gather, associate Q-values with the executed actions
            action_probs = tf.gather(tf.reshape(next_Q, [-1]), idx_flattened)

            Y = rewards + self.params.gamma * action_probs * np.logical_not(dones)

            # calculate Q(s,a)
            q_values = self.main_model(tf.convert_to_tensor(states, dtype=tf.float32))

            # get the q-values which is associated with actually taken actions in a game
            actions_one_hot = tf.one_hot(actions, self.num_action, 1.0, 0.0)
            action_probs = tf.reduce_sum(actions_one_hot * q_values, reduction_indices=-1)

            if self.params.loss_fn == "huber_loss":
                # use huber loss
                loss = huber_loss(tf.subtract(Y, action_probs))
                batch_loss = loss
            elif self.params.loss_fn == "MSE":
                # use MSE
                batch_loss = tf.squared_difference(Y, action_probs)
                loss = tf.reduce_mean(batch_loss)
            else:
                assert False

        # get gradients
        grads = tape.gradient(loss, self.main_model.trainable_weights)

        # clip gradients
        if self.params.grad_clip_flg == "by_value":
            grads = [ClipIfNotNone(grad, -1., 1.) for grad in grads]
        elif self.params.grad_clip_flg == "norm":
            grads, _ = tf.clip_by_global_norm(grads, 5.0)

        # apply processed gradients to the network
        self.optimizer.apply_gradients(zip(grads, self.main_model.trainable_weights))

        # for log purpose
        for index, grad in enumerate(grads):
            tf.contrib.summary.histogram("layer_grad_{}".format(index), grad, step=self.index_episode)
        tf.contrib.summary.scalar("loss", loss, step=self.index_episode)
        tf.contrib.summary.histogram("next_Q", next_Q, step=self.index_episode)
        tf.contrib.summary.scalar("mean_q_value", tf.math.reduce_mean(next_Q), step=self.index_episode)
        tf.contrib.summary.scalar("var_q_value", tf.math.reduce_variance(next_Q), step=self.index_episode)
        tf.contrib.summary.scalar("max_q_value", tf.reduce_max(next_Q), step=self.index_episode)

        return loss, batch_loss