import numpy as np
import tensorflow as tf
from tf_rl.common.utils import huber_loss, ClipIfNotNone


class DQN:
    """
    DQN model which is reusable for duelling dqn as well
    We only normalise a state/next_state pixels by 255 when we feed them into a model.
    Replay buffer stores them as np.int8 because of memory issue this is also stated in OpenAI Baselines wrapper.
    """
    def __init__(self, env_type, main_model, target_model, num_action, params, checkpoint_dir="../logs/models/DQN/"):
        self.env_type = env_type
        self.num_action = num_action
        self.params = params
        self.main_model = main_model(env_type, num_action)
        self.target_model = target_model(env_type, num_action)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        # self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.index_episode = 0

        # TF: checkpoint vs Saver => https://stackoverflow.com/questions/53569622/difference-between-tf-train-checkpoint-and-tf-train-saver
        self.checkpoint_dir = checkpoint_dir
        self.check_point = tf.train.Checkpoint(optimizer=self.optimizer,
                                               model=self.main_model,
                                               optimizer_step=tf.train.get_or_create_global_step())
        self.manager = tf.train.CheckpointManager(self.check_point, checkpoint_dir, max_to_keep=3)

    def predict(self, state):
        if self.env_type == "Atari":
            state = state.astype('float32') / 255.
        return self.main_model(tf.convert_to_tensor(state[None,:], dtype=tf.float32)).numpy()[0]

    def update(self, states, actions, rewards, next_states, dones):
        if self.env_type == "Atari":
            states, next_states = states.astype('float32') / 255., next_states.astype('float32') / 255.
        with tf.GradientTape() as tape:
            # make sure to fit all process to compute gradients within this Tape context!!

            # calculate target: R + gamma * max_a Q(s',a')
            next_Q = self.target_model(tf.convert_to_tensor(next_states, dtype=tf.float32))
            Y = rewards + self.params.gamma * np.max(next_Q, axis=-1).flatten() * np.logical_not(dones)

            # calculate Q(s,a)
            q_values = self.main_model(tf.convert_to_tensor(states, dtype=tf.float32))

            # get the q-values which is associated with actually taken actions in a game
            actions_one_hot = tf.one_hot(actions, self.num_action, 1.0, 0.0)
            action_probs = tf.reduce_sum(actions_one_hot*q_values, reduction_indices=1)

            if self.params.loss_fn == "huber_loss":
                # use huber loss
                batch_loss = huber_loss(tf.squared_difference(Y, action_probs))
                loss = tf.reduce_mean(batch_loss)
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
        elif self.params.grad_clip_flg == "None":
            pass

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