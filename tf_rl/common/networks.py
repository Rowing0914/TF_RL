import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

L2 = tf.keras.regularizers.l2(1e-2)
KERNEL_INIT = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
XAVIER_INIT = tf.contrib.layers.xavier_initializer()


class Nature_DQN(tf.keras.Model):
    def __init__(self, num_action):
        super(Nature_DQN, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=8, strides=8, activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, activation='relu')
        self.flat = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(512, activation='relu')
        self.pred = tf.keras.layers.Dense(num_action, activation='linear')

    @tf.contrib.eager.defun(autograph=False)
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flat(x)
        x = self.fc1(x)
        return self.pred(x)


class CartPole(tf.keras.Model):
    def __init__(self, num_action):
        super(CartPole, self).__init__()
        self.dense1 = tf.keras.layers.Dense(16, activation='relu')
        self.dense2 = tf.keras.layers.Dense(16, activation='relu')
        self.dense3 = tf.keras.layers.Dense(16, activation='relu')
        self.pred = tf.keras.layers.Dense(num_action, activation='linear')

    @tf.contrib.eager.defun(autograph=False)
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.pred(x)


class Duelling_atari(tf.keras.Model):
    def __init__(self, num_action, duelling_type="avg"):
        super(Duelling_atari, self).__init__()
        self.duelling_type = duelling_type
        self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=8, strides=8, activation='relu', kernel_regularizer=L2,
                                            bias_regularizer=L2)
        self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, activation='relu', kernel_regularizer=L2,
                                            bias_regularizer=L2)
        self.conv3 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, activation='relu', kernel_regularizer=L2,
                                            bias_regularizer=L2)
        self.flat = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=L2, bias_regularizer=L2)
        self.q_value = tf.keras.layers.Dense(num_action, activation='linear', kernel_regularizer=L2,
                                             bias_regularizer=L2)
        self.v_value = tf.keras.layers.Dense(1, activation='linear', kernel_regularizer=L2, bias_regularizer=L2)

    @tf.contrib.eager.defun(autograph=False)
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flat(x)
        x = self.fc1(x)
        q_value = self.q_value(x)
        v_value = self.v_value(x)

        if self.duelling_type == "avg":
            # Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-Avg_a(A(s,a;theta)))
            output = tf.math.add(v_value, tf.math.subtract(q_value, tf.reduce_mean(q_value)))
        elif self.duelling_type == "max":
            # Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-max_a(A(s,a;theta)))
            output = tf.math.add(v_value, tf.math.subtract(q_value, tf.math.reduce_max(q_value)))
        elif self.duelling_type == "naive":
            # Q(s,a;theta) = V(s;theta) + A(s,a;theta)
            output = tf.math.add(v_value, q_value)
        else:
            output = 0  # defun does not accept the variable may not be intialised, so that temporarily initialise it
            assert False, "dueling_type must be one of {'avg','max','naive'}"
        return output


class Duelling_cartpole(tf.keras.Model):
    def __init__(self, num_action, duelling_type="avg"):
        super(Duelling_cartpole, self).__init__()
        self.duelling_type = duelling_type
        self.dense1 = tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=L2, bias_regularizer=L2, )
        self.dense2 = tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=L2, bias_regularizer=L2, )
        self.dense3 = tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=L2, bias_regularizer=L2, )
        self.q_value = tf.keras.layers.Dense(num_action, activation='linear', kernel_regularizer=L2,
                                             bias_regularizer=L2, )
        self.v_value = tf.keras.layers.Dense(1, activation='linear', kernel_regularizer=L2, bias_regularizer=L2, )

    @tf.contrib.eager.defun(autograph=False)
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        q_value = self.q_value(x)
        v_value = self.v_value(x)

        if self.duelling_type == "avg":
            # Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-Avg_a(A(s,a;theta)))
            output = tf.math.add(v_value, tf.math.subtract(q_value, tf.reduce_mean(q_value)))
        elif self.duelling_type == "max":
            # Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-max_a(A(s,a;theta)))
            output = tf.math.add(v_value, tf.math.subtract(q_value, tf.math.reduce_max(q_value)))
        elif self.duelling_type == "naive":
            # Q(s,a;theta) = V(s;theta) + A(s,a;theta)
            output = tf.math.add(v_value, q_value)
        else:
            output = 0  # defun does not accept the variable may not be intialised, so that temporarily initialise it
            assert False, "dueling_type must be one of {'avg','max','naive'}"
        return output


class DDPG_Actor(tf.keras.Model):
    def __init__(self, num_action=1):
        super(DDPG_Actor, self).__init__()
        self.dense1 = tf.keras.layers.Dense(400, activation='relu', kernel_initializer=KERNEL_INIT)
        self.dense2 = tf.keras.layers.Dense(300, activation='relu', kernel_initializer=KERNEL_INIT)
        self.pred = tf.keras.layers.Dense(num_action, activation='tanh', kernel_initializer=KERNEL_INIT)

    @tf.contrib.eager.defun(autograph=False)
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        pred = self.pred(x)
        return pred


class DDPG_Critic(tf.keras.Model):
    def __init__(self, output_shape):
        super(DDPG_Critic, self).__init__()
        self.dense1 = tf.keras.layers.Dense(400, activation='relu', kernel_regularizer=L2, bias_regularizer=L2,
                                            kernel_initializer=KERNEL_INIT)
        self.dense2 = tf.keras.layers.Dense(300, activation='relu', kernel_regularizer=L2, bias_regularizer=L2,
                                            kernel_initializer=KERNEL_INIT)
        self.pred = tf.keras.layers.Dense(output_shape, activation='linear', kernel_regularizer=L2, bias_regularizer=L2,
                                          kernel_initializer=KERNEL_INIT)

    @tf.contrib.eager.defun(autograph=False)
    def call(self, obs, act):
        x = self.dense1(obs)
        x = self.dense2(tf.concat([x, act], axis=-1))
        pred = self.pred(x)
        return pred


class BatchNorm_DDPG_Actor(tf.keras.Model):
    def __init__(self, num_action=1):
        super(BatchNorm_DDPG_Actor, self).__init__()
        self.dense1 = tf.keras.layers.Dense(400, activation='relu', kernel_initializer=KERNEL_INIT)
        self.batch1 = tf.keras.layers.BatchNormalization()
        self.dense2 = tf.keras.layers.Dense(300, activation='relu', kernel_initializer=KERNEL_INIT)
        self.batch2 = tf.keras.layers.BatchNormalization()
        self.pred = tf.keras.layers.Dense(num_action, activation='tanh', kernel_initializer=KERNEL_INIT)

    @tf.contrib.eager.defun(autograph=False)
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.batch1(x)
        x = self.dense2(x)
        x = self.batch2(x)
        pred = self.pred(x)
        return pred


class BatchNorm_DDPG_Critic(tf.keras.Model):
    def __init__(self, output_shape):
        super(BatchNorm_DDPG_Critic, self).__init__()
        self.dense1 = tf.keras.layers.Dense(400, activation='relu', kernel_regularizer=L2, bias_regularizer=L2,
                                            kernel_initializer=KERNEL_INIT)
        self.batch1 = tf.keras.layers.BatchNormalization()
        self.dense2 = tf.keras.layers.Dense(300, activation='relu', kernel_regularizer=L2, bias_regularizer=L2,
                                            kernel_initializer=KERNEL_INIT)
        self.batch2 = tf.keras.layers.BatchNormalization()
        self.pred = tf.keras.layers.Dense(output_shape, activation='linear', kernel_regularizer=L2, bias_regularizer=L2,
                                          kernel_initializer=KERNEL_INIT)

    @tf.contrib.eager.defun(autograph=False)
    def call(self, obs, act):
        x = self.dense1(obs)
        x = self.batch1(x)
        x = self.dense2(tf.concat([x, act], axis=-1))
        x = self.batch2(x)
        pred = self.pred(x)
        return pred


class self_rewarding_DDPG_Actor(tf.keras.Model):
    def __init__(self, num_action=1):
        super(self_rewarding_DDPG_Actor, self).__init__()
        self.dense1 = tf.keras.layers.Dense(400, activation='relu', kernel_initializer=KERNEL_INIT)
        self.batch1 = tf.keras.layers.BatchNormalization()
        self.dense2 = tf.keras.layers.Dense(300, activation='relu', kernel_initializer=KERNEL_INIT)
        self.batch2 = tf.keras.layers.BatchNormalization()
        self.pred = tf.keras.layers.Dense(num_action, activation='tanh', kernel_initializer=KERNEL_INIT)
        self.reward = tf.keras.layers.Dense(1, activation='tanh', kernel_initializer=KERNEL_INIT)

    @tf.contrib.eager.defun(autograph=False)
    def call(self, inputs):
        x = self.dense1(inputs)
        # x = self.batch1(x)
        x = self.dense2(x)
        x = self.batch2(x)
        pred = self.pred(x)
        reward = self.reward(x)
        return pred, reward


class HER_Actor(tf.keras.Model):
    """
    In paper, it's saying that all layers consist of 64 neurons...
    but in OpenAI her implementation, they used 256. so I'll stick with 256

    """

    def __init__(self, num_action=1):
        super(HER_Actor, self).__init__()
        self.dense1 = tf.keras.layers.Dense(256, activation='relu', kernel_initializer=KERNEL_INIT)
        self.dense2 = tf.keras.layers.Dense(256, activation='relu', kernel_initializer=KERNEL_INIT)
        self.dense3 = tf.keras.layers.Dense(256, activation='relu', kernel_initializer=KERNEL_INIT)
        self.pred = tf.keras.layers.Dense(num_action, activation='tanh', kernel_initializer=KERNEL_INIT)

    @tf.contrib.eager.defun(autograph=False)
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        pred = self.pred(x)
        return pred


class HER_Critic(tf.keras.Model):
    """
    In paper, it's saying that all layers consist of 64 neurons...
    but in OpenAI her implementation, they used 256. so I'll stick with 256

    """

    def __init__(self, output_shape):
        super(HER_Critic, self).__init__()
        self.dense1 = tf.keras.layers.Dense(256, activation='relu', kernel_initializer=KERNEL_INIT)
        self.dense2 = tf.keras.layers.Dense(256, activation='relu', kernel_initializer=KERNEL_INIT)
        self.dense3 = tf.keras.layers.Dense(256, activation='relu', kernel_initializer=KERNEL_INIT)
        self.pred = tf.keras.layers.Dense(output_shape, activation='linear', kernel_initializer=KERNEL_INIT)

    @tf.contrib.eager.defun(autograph=False)
    def call(self, inputs, act):
        # _input is already concatenated of obs and g
        x = self.dense1(inputs)
        x = self.dense2(tf.concat([x, act], axis=-1))
        x = self.dense3(x)
        pred = self.pred(x)
        return pred


class SAC_Actor(tf.keras.Model):
    """
    Policy network: Gaussian Policy.
    It outputs Mean and Std with the size of number of actions.
    And we sample from Normal dist upon resulting Mean&Std

    In Haarnoja's implementation, he uses 100 neurons for hidden layers... so it's up to you!!
    """

    def __init__(self, num_action=1):
        super(SAC_Actor, self).__init__()
        self.LOG_SIG_MAX = 2
        self.LOG_SIG_MIN = -20

        self.dense1 = tf.keras.layers.Dense(256, activation='relu', kernel_initializer=XAVIER_INIT)
        self.dense2 = tf.keras.layers.Dense(256, activation='relu', kernel_initializer=XAVIER_INIT)
        self.mean = tf.keras.layers.Dense(num_action, activation='linear', kernel_initializer=XAVIER_INIT)
        self.std = tf.keras.layers.Dense(num_action, activation='linear', kernel_initializer=XAVIER_INIT)

    @tf.contrib.eager.defun(autograph=False)
    def call(self, inputs):
        """
        As mentioned in the topic of `policy evaluation` at sec5.2(`ablation study`) in the paper,
        for evaluation phase, using a deterministic action(choosing the mean of the policy dist) works better than
        stochastic one(Gaussian Policy). So that we need to output three different values. I know it's kind of weird design..
        """
        x = self.dense1(inputs)
        x = self.dense2(x)
        mean = self.mean(x)
        std = self.std(x)
        std = tf.clip_by_value(std, self.LOG_SIG_MIN, self.LOG_SIG_MAX)
        std = tf.math.exp(std)
        dist = tfd.Normal(loc=mean, scale=std)
        # dist = tfd.MultivariateNormalDiag(loc=mean, scale_diag=std)
        x = dist.sample()
        action = tf.keras.activations.tanh(x)
        log_prob = dist.log_prob(x)
        log_prob -= tf.math.log(1. - tf.math.square(action) + 1e-6)
        log_prob = tf.math.reduce_sum(log_prob, 1, keep_dims=True)
        return action, log_prob, tf.keras.activations.tanh(mean)


class SAC_Critic(tf.keras.Model):
    """
    It contains two Q-network. And the usage of two Q-functions improves performance by reducing overestimation bias.
    """

    def __init__(self, output_shape):
        super(SAC_Critic, self).__init__()
        # Q1 architecture
        self.dense1 = tf.keras.layers.Dense(256, activation='relu', kernel_initializer=XAVIER_INIT)
        self.dense2 = tf.keras.layers.Dense(256, activation='relu', kernel_initializer=XAVIER_INIT)
        self.Q1 = tf.keras.layers.Dense(output_shape, activation='linear', kernel_initializer=XAVIER_INIT)

        # Q2 architecture
        self.dense3 = tf.keras.layers.Dense(256, activation='relu', kernel_initializer=XAVIER_INIT)
        self.dense4 = tf.keras.layers.Dense(256, activation='relu', kernel_initializer=XAVIER_INIT)
        self.Q2 = tf.keras.layers.Dense(output_shape, activation='linear', kernel_initializer=XAVIER_INIT)

    @tf.contrib.eager.defun(autograph=False)
    def call(self, obs, act):
        x1 = self.dense1(obs)
        x1 = self.dense2(tf.concat([x1, act], axis=-1))
        Q1 = self.Q1(x1)

        x2 = self.dense3(obs)
        x2 = self.dense4(tf.concat([x2, act], axis=-1))
        Q2 = self.Q2(x2)
        return Q1, Q2

    # @tf.contrib.eager.defun(autograph=False)
    # def call(self, obs, act):
    #     _concat = tf.concat([obs, act], axis=-1)
    #     x1 = self.dense1(_concat)
    #     x1 = self.dense2(x1)
    #     Q1 = self.Q1(x1)
    #
    #     x2 = self.dense3(_concat)
    #     x2 = self.dense4(x2)
    #     Q2 = self.Q2(x2)
    #     return Q1, Q2


class TRPO_Policy(tf.keras.Model):
    """
    TRPO Policy network
    """

    def __init__(self, output_shape):
        super(TRPO_Policy, self).__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='tanh', kernel_initializer=KERNEL_INIT)
        self.dense2 = tf.keras.layers.Dense(128, activation='tanh', kernel_initializer=KERNEL_INIT)
        self.mean = tf.keras.layers.Dense(output_shape, activation='linear', kernel_initializer=KERNEL_INIT)
        self.std = tf.get_variable('sigma', (1, output_shape), tf.float32, tf.constant_initializer(0.6))

    @tf.contrib.eager.defun(autograph=False)
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        mean = self.mean(x)
        return mean, self.std


class TRPO_Value(tf.keras.Model):
    """
    TRPO State Value network
    """

    def __init__(self, output_shape):
        super(TRPO_Value, self).__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='tanh', kernel_regularizer=L2, bias_regularizer=L2,
                                            kernel_initializer=KERNEL_INIT)
        self.dense2 = tf.keras.layers.Dense(128, activation='tanh', kernel_regularizer=L2, bias_regularizer=L2,
                                            kernel_initializer=KERNEL_INIT)
        self.pred = tf.keras.layers.Dense(output_shape, activation='linear', kernel_regularizer=L2, bias_regularizer=L2,
                                          kernel_initializer=KERNEL_INIT)

    @tf.contrib.eager.defun(autograph=False)
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        pred = self.pred(x)
        return pred
