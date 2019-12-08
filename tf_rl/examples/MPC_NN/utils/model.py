import tensorflow as tf


class Model(tf.Module):
    def __init__(self, state_shape, name="DynamicsModel"):
        super(Model, self).__init__(name=name)
        self.dense1 = tf.keras.layers.Dense(500, activation="relu")
        self.dense2 = tf.keras.layers.Dense(500, activation="relu")
        self.dense3 = tf.keras.layers.Dense(state_shape, activation="linear")

    def __call__(self, state, action):
        x = tf.concat([state, action], axis=-1)
        x = self.dense1(x)
        x = self.dense2(x)
        state_delta = self.dense3(x)
        return state_delta


class DynamicsModel:
    def __init__(self, state_shape):
        self.model = Model(state_shape)
        self.optimiser = tf.compat.v1.train.AdamOptimizer()
        self.mean_delta = tf.random.normal(shape=[state_shape], dtype=tf.float32)
        self.std_delta = tf.random.normal(shape=[state_shape], dtype=tf.float32)

    @tf.function
    def predict(self, state, action):
        pred_state_delta = self.model(state, action)
        pred_state_delta = tf.cast(pred_state_delta, dtype=tf.float32)

        # perturb the predicted delta(s_t+1 - s_t) and produce the next state using the current state
        return pred_state_delta * self.std_delta + self.mean_delta + state

    def update_mean_std(self, mean_delta, std_delta):
        """ Update the params in the sampling distribution of predicted states """
        self.mean_delta = mean_delta
        self.std_delta = std_delta

    @tf.function
    def update(self, states, actions, deltas):
        with tf.GradientTape() as tape:
            predict = self.predict(states, actions)
            loss = tf.compat.v1.losses.mean_squared_error(deltas, predict)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimiser.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss
