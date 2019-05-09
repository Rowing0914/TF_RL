import numpy as np
import tensorflow as tf
from tf_rl.common.utils import ClipIfNotNone, AnnealingSchedule

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
        # self.learning_rate = AnnealingSchedule(start=1e-3, end=1e-5, decay_steps=params.decay_steps, decay_type="linear") # learning rate decay!!
        # self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate.get_value())

        self.learning_rate = AnnealingSchedule(start=0.0025, end=0.00025, decay_steps=params.decay_steps,
                                               decay_type="linear")  # learning rate decay!!
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate.get_value(), 0.99, 0.0, 1e-6)

        # TF: checkpoint vs Saver => https://stackoverflow.com/questions/53569622/difference-between-tf-train-checkpoint-and-tf-train-saver
        self.checkpoint_dir = checkpoint_dir
        self.check_point = tf.train.Checkpoint(optimizer=self.optimizer,
                                               model=self.main_model,
                                               optimizer_step=tf.train.get_or_create_global_step())
        self.manager = tf.train.CheckpointManager(self.check_point, checkpoint_dir, max_to_keep=3)

    def predict(self, state):
        if self.env_type == "Atari":
            state = np.array(state).astype('float32') / 255.
        return self.main_model(tf.convert_to_tensor(state[None,:], dtype=tf.float32)).numpy()[0]

    def update(self, states, actions, rewards, next_states, dones):
        # let's keep this for debug purpose!!
        # if you feel that the agent does not keep up with the global time-step, pls open this!
        # print("===== UPDATE ===== Train Step:{}".format(tf.train.get_global_step()))

        # get the current global-timestep
        self.index_timestep = tf.train.get_global_step()

        if self.env_type == "Atari":
            states, next_states = np.array(states).astype('float32') / 255., np.array(next_states).astype('float32') / 255.

        # ===== make sure to fit all process to compute gradients within this Tape context!! =====
        with tf.GradientTape() as tape:
            next_Q = self.target_model(tf.convert_to_tensor(next_states, dtype=tf.float32))
            Y = rewards + self.params.gamma * np.max(next_Q, axis=-1).flatten() * (1. - tf.cast(dones, tf.float32))
            Y = tf.stop_gradient(Y)

            # calculate Q(s,a)
            q_values = self.main_model(tf.convert_to_tensor(states, dtype=tf.float32))

            # get the q-values which is associated with actually taken actions in a game
            actions_one_hot = tf.one_hot(actions, self.num_action, 1.0, 0.0)
            chosen_q = tf.reduce_sum(actions_one_hot*q_values, reduction_indices=1)

            if self.params.loss_fn == "huber_loss":
                # use huber loss
                batch_loss = tf.losses.huber_loss(Y, chosen_q, reduction=tf.losses.Reduction.NONE)
                loss = tf.reduce_mean(batch_loss)
            elif self.params.loss_fn == "MSE":
                # use MSE
                batch_loss = tf.squared_difference(Y, chosen_q)
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
        self.optimizer.apply_gradients(zip(grads, self.main_model.trainable_weights), global_step=self.index_timestep)

        # for log purpose
        for index, grad in enumerate(grads):
            tf.contrib.summary.histogram("layer_grad_{}".format(index), grad, step=self.index_timestep)
        tf.contrib.summary.scalar("loss", loss, step=self.index_timestep)
        tf.contrib.summary.histogram("next_Q", next_Q, step=self.index_timestep)
        tf.contrib.summary.histogram("Y(target)", Y, step=self.index_timestep)
        tf.contrib.summary.scalar("mean_q_value", tf.math.reduce_mean(next_Q), step=self.index_timestep)
        tf.contrib.summary.scalar("var_q_value", tf.math.reduce_variance(next_Q), step=self.index_timestep)
        tf.contrib.summary.scalar("max_q_value", tf.reduce_max(next_Q), step=self.index_timestep)
        tf.contrib.summary.scalar("learning_rate", self.learning_rate.get_value(), step=self.index_timestep)

        return loss, batch_loss



class DQN_new:
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
        # self.learning_rate = AnnealingSchedule(start=1e-3, end=1e-5, decay_steps=params.decay_steps, decay_type="linear") # learning rate decay!!
        # self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate.get_value())

        self.learning_rate = AnnealingSchedule(start=0.0025, end=0.00025, decay_steps=params.decay_steps,
                                               decay_type="linear")  # learning rate decay!!
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate.get_value(), 0.99, 0.0, 1e-6)

        # TF: checkpoint vs Saver => https://stackoverflow.com/questions/53569622/difference-between-tf-train-checkpoint-and-tf-train-saver
        self.checkpoint_dir = checkpoint_dir
        self.check_point = tf.train.Checkpoint(optimizer=self.optimizer,
                                               model=self.main_model,
                                               optimizer_step=tf.train.get_or_create_global_step())
        self.manager = tf.train.CheckpointManager(self.check_point, checkpoint_dir, max_to_keep=3)

    def predict(self, state):
        if self.env_type == "Atari":
            state = np.array(state).astype('float32') / 255.
        return self.main_model(tf.convert_to_tensor(state[None,:], dtype=tf.float32)).numpy()[0]

    def update(self, states, actions, rewards, next_states, dones):
        # let's keep this for debug purpose!!
        # if you feel that the agent does not keep up with the global time-step, pls open this!
        # print("===== UPDATE ===== Train Step:{}".format(tf.train.get_global_step()))

        # get the current global-timestep
        self.index_timestep = tf.train.get_global_step()

        if self.env_type == "Atari":
            states, next_states = np.array(states).astype('float32') / 255., np.array(next_states).astype('float32') / 255.

        # ===== make sure to fit all process to compute gradients within this Tape context!! =====
        with tf.GradientTape() as tape:
            next_Q = self.target_model(tf.convert_to_tensor(next_states, dtype=tf.float32))
            Y = rewards + self.params.gamma * np.max(next_Q, axis=-1).flatten() * (1. - tf.cast(dones, tf.float32))
            Y = tf.stop_gradient(Y)

            # calculate Q(s,a)
            q_values = self.main_model(tf.convert_to_tensor(states, dtype=tf.float32))

            # at this point, instead of getting only q-values associated wit taken actions
            # we retain all values except that we update q-values associated wit taken actions by "Y"
            # Shape of Q-values matrix: (32,2)
            target_values = tf.one_hot(actions, self.num_action, 1.0, 0.0)*np.array([Y,Y]).T + tf.one_hot(actions, self.num_action, 0.0, 1.0)*q_values
            assert tf.math.equal(target_values, q_values).numpy().all() == False, "Your target values are not updated correctly"

            if self.params.loss_fn == "huber_loss":
                # use huber loss
                batch_loss = tf.losses.huber_loss(target_values, q_values, reduction=tf.losses.Reduction.NONE)
                loss = tf.reduce_mean(batch_loss)
            elif self.params.loss_fn == "MSE":
                # use MSE
                batch_loss = tf.squared_difference(target_values, q_values)
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
        self.optimizer.apply_gradients(zip(grads, self.main_model.trainable_weights), global_step=self.index_timestep)

        # for log purpose
        for index, grad in enumerate(grads):
            tf.contrib.summary.histogram("layer_grad_{}".format(index), grad, step=self.index_timestep)
        tf.contrib.summary.scalar("loss", loss, step=self.index_timestep)
        tf.contrib.summary.histogram("next_Q", next_Q, step=self.index_timestep)
        tf.contrib.summary.histogram("Y(target)", Y, step=self.index_timestep)
        tf.contrib.summary.scalar("mean_q_value", tf.math.reduce_mean(next_Q), step=self.index_timestep)
        tf.contrib.summary.scalar("var_q_value", tf.math.reduce_variance(next_Q), step=self.index_timestep)
        tf.contrib.summary.scalar("max_q_value", tf.reduce_max(next_Q), step=self.index_timestep)
        tf.contrib.summary.scalar("learning_rate", self.learning_rate.get_value(), step=self.index_timestep)

        return loss, batch_loss