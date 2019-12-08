import tensorflow as tf

KERNEL_INIT = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
XAVIER_INIT = tf.contrib.layers.xavier_initializer()
DECAY_PARAMS = [0.000025, 0.00005, 0.000075, 0.000075, 0.0001]  # default param in halfcheetah
ENSEMBLE_PREFIX = "ensemble_"
SWISH = lambda x: x * tf.sigmoid(x)

class Model(tf.Module):
    def __init__(self, in_features, out_features, id):
        super(Model, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        with tf.name_scope(str(ENSEMBLE_PREFIX + "{}".format(id))):
            self.max_logvar = tf.Variable(tf.ones(shape=(1, out_features), dtype=tf.float32) / 2.0, name="max_logvar")
            self.min_logvar = tf.Variable(- tf.ones(shape=(1, out_features), dtype=tf.float32) * 10.0,
                                          name="min_logvar")

            self.dense1 = tf.keras.layers.Dense(in_features, kernel_initializer=KERNEL_INIT)
            self.dense2 = tf.keras.layers.Dense(200, kernel_initializer=KERNEL_INIT)
            self.dense3 = tf.keras.layers.Dense(200, kernel_initializer=KERNEL_INIT)
            self.dense4 = tf.keras.layers.Dense(200, kernel_initializer=KERNEL_INIT)
            self.dense5 = tf.keras.layers.Dense(out_features * 2, kernel_initializer=KERNEL_INIT)

    def __call__(self, inputs, ret_logvar=False):
        x = self.dense1(inputs)
        x = SWISH(x)
        x = self.dense2(x)
        x = SWISH(x)
        x = self.dense3(x)
        x = SWISH(x)
        x = self.dense4(x)
        x = SWISH(x)
        outputs = self.dense5(x)
        mean, logvar = tf.split(outputs, 2, axis=-1)

        logvar = self.max_logvar - tf.nn.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + tf.nn.softplus(logvar - self.min_logvar)

        if ret_logvar:
            return mean, logvar

        return mean, tf.math.exp(logvar)


class HalfCheetahModel(tf.Module):
    def __init__(self, ensemble_size, in_features, out_features, decay_params=DECAY_PARAMS):
        super(HalfCheetahModel, self).__init__()

        self.inputs_mu = tf.constant(tf.zeros(in_features))
        self.inputs_sigma = tf.constant(tf.zeros(in_features))
        self.ensemble_size = ensemble_size
        self.decay_params = decay_params

        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)

        # instantiate the ensembling dynamics models
        for idx in range(ensemble_size):
            setattr(self, "model_{}".format(idx), Model(in_features=in_features, out_features=out_features, id=idx))

        self.reduce_params_among_models()

    def __call__(self, inputs, ret_logvar=False):
        means, logvars = [], []

        # Scaling inputs
        inputs = (inputs - self.inputs_mu) / self.inputs_sigma

        for idx in range(self.ensemble_size):
            mean, logvar = getattr(self, "model_{}".format(idx))(inputs)
            means.append(mean)
            logvars.append(logvar)

        # average the result across the ensembling models to get an outcome among them
        return tf.math.reduce_mean(means, axis=0), tf.math.reduce_mean(logvars, axis=0)

    def fit_input_stats(self, data):
        """ update the scaling factors of inputs """
        self.inputs_mu = tf.math.reduce_mean(data, axis=0, keepdims=True)
        sigma = tf.math.reduce_std(data, axis=0, keepdims=True)
        sigma = tf.cast(sigma, dtype=tf.float32)
        self.inputs_sigma = tf.where(sigma < 1e-12, tf.ones_like(sigma, dtype=tf.float32), sigma)

    def reduce_params_among_models(self):
        """ collect necessary information from each model and summarise them as properties of this class """
        max_logvars, min_logvars = [], []
        for idx in range(self.ensemble_size):
            max_logvars.append(getattr(self, "model_{}".format(idx)).max_logvar)
            min_logvars.append(getattr(self, "model_{}".format(idx)).min_logvar)

        self.max_logvar = tf.math.reduce_mean(max_logvars, axis=0)
        self.min_logvar = tf.math.reduce_mean(min_logvars, axis=0)

    def reduce_L2_losses_among_models(self):
        """ calculate the averaged magnitude of weight decay among all ensembling models
            Note:
                this has to be called after you fit the data. otherwise, the weights in the layers aren't initialised.
                And you would get an error.
        """
        l2_losses = []
        for idx in range(self.ensemble_size):
            weights = getattr(self, "model_{}".format(idx)).trainable_variables
            decay = self._compute_decays(weights=weights)
            l2_losses.append(decay)

        self.decay_losses = tf.math.reduce_mean(l2_losses, axis=0)

    def _compute_decays(self, weights):
        """ Manually compute the magnitude of weight decay """
        decay = 0.0
        for idx, weight in enumerate(weights):
            try:
                var_name = weight.name.split("/")[1]
                if var_name == "kernel:0":
                    _coef = self.decay_params[idx]
                    decay += _coef * tf.math.reduce_sum(weight ** 2) / 2.0
            except:
                pass
        return decay
