from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from eager.optimizers_eager import CEMOptimizer
from eager.dynamics_models_eager import HalfCheetahModel


class Controller:
    def __init__(self, *args, **kwargs):
        """Creates class instance.
        """
        pass

    def train(self, obs_trajs, acs_trajs, rews_trajs):
        """Trains this controller using lists of trajectories.
        """
        raise NotImplementedError("Must be implemented in subclass.")

    def reset(self):
        """Resets this controller.
        """
        raise NotImplementedError("Must be implemented in subclass.")

    def act(self, obs, t, has_been_trained, get_pred_cost=False):
        """Performs an action.
        """
        raise NotImplementedError("Must be implemented in subclass.")

    def dump_logs(self, primary_logdir, iter_logdir):
        """Dumps logs into primary log directory and per-train iteration log directory.
        """
        raise NotImplementedError("Must be implemented in subclass.")


def shuffle_rows(arr):
    idxs = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
    return arr[np.arange(arr.shape[0])[:, None], idxs]


class MPC(Controller):
    optimizers = {"CEM": CEMOptimizer}

    def __init__(self, env):
        super().__init__(env)

        self.per = 1
        self.prop_mode = "TSinf"
        self.npart = 20
        self.ign_var = False
        self.opt_mode = "CEM"
        self.plan_hor = 30
        self.epochs = 10
        self.batch_size = 32
        self.ensemble_size = 5
        self.save_all_models = False
        self.log_traj_preds = False
        self.log_particles = False

        self.dO, self.dU = env.observation_space.shape[0], env.action_space.shape[0]
        self.ac_ub, self.ac_lb = env.action_space.high, env.action_space.low
        # self.update_fns = params.get("update_fns", [])  #TODO: only reacher uses this part

        # TODO: these processes of obs/next_obs are defined in each env. so when MuJoCo is available, come back here!!
        """
        self.obs_preproc = params.prop_cfg.get("obs_preproc", lambda obs: obs)
        self.obs_postproc = params.prop_cfg.get("obs_postproc", lambda obs, model_out: model_out)
        self.obs_postproc2 = params.prop_cfg.get("obs_postproc2", lambda next_obs: next_obs)
        self.targ_proc = params.prop_cfg.get("targ_proc", lambda obs, next_obs: next_obs)
        """
        self.obs_preproc = lambda obs: obs
        self.obs_postproc = lambda obs, model_out: model_out
        self.obs_postproc2 = lambda next_obs: next_obs
        self.targ_proc = lambda obs, next_obs: next_obs
        self.obs_cost_fn = lambda obs: -obs[:, 0]
        self.ac_cost_fn = lambda action: 0.1 * tf.reduce_sum(action ** 2, axis=1)

        # Create action sequence optimizer
        self.optimizer = CEMOptimizer(
            sol_dim=self.plan_hor * self.dU,
            lower_bound=np.tile(self.ac_lb, [self.plan_hor]),
            upper_bound=np.tile(self.ac_ub, [self.plan_hor]),
            cost_function=self._compile_cost,
            max_iters=5,
            popsize=500,
            num_elites=50,
            alpha=0.1
        )

        # Controller state variables
        self.ac_buf = np.array([]).reshape(0, self.dU)
        self.prev_sol = np.tile((self.ac_lb + self.ac_ub) / 2, [self.plan_hor]).astype(np.float32)
        self.init_var = np.tile(np.square(self.ac_ub - self.ac_lb) / 16, [self.plan_hor]).astype(np.float32)
        self.train_in = np.array([]).reshape(0, self.dU + self.obs_preproc(np.zeros([1, self.dO])).shape[-1]).astype(
            np.float32)
        self.train_targs = np.array([]).reshape(
            0, self.targ_proc(np.zeros([1, self.dO]), np.zeros([1, self.dO])).shape[-1]
        ).astype(np.float32)

        print("Created an MPC controller, prop mode %s, %d particles. " % (self.prop_mode, self.npart) +
              ("Ignoring variance." if self.ign_var else ""))

        if self.save_all_models:
            print("Controller will save all models. (Note: This may be memory-intensive.")
        if self.log_particles:
            print("Controller is logging particle predictions (Note: This may be memory-intensive).")
            self.pred_particles = []
        elif self.log_traj_preds:
            print("Controller is logging trajectory prediction statistics (mean+var).")
            self.pred_means, self.pred_vars = [], []
        else:
            print("Trajectory prediction logging is disabled.")

        # Set up the dynamics model
        self.model = HalfCheetahModel(in_features=self.dO + self.dU,
                                      out_features=self.dO,
                                      ensemble_size=self.ensemble_size)

    def train(self, obs_trajs, acs_trajs, rews_trajs):
        """Trains the internal model of this controller. Once trained,
        this controller switches from applying random actions to using MPC.

        Arguments:
            obs_trajs: A list of observation matrices, observations in rows.
            acs_trajs: A list of action matrices, actions in rows.
            rews_trajs: A list of reward arrays.

        Returns: None.
        """

        # Construct new training points and add to training set
        new_train_in, new_train_targs = [], []
        for obs, acs in zip(obs_trajs, acs_trajs):
            new_train_in.append(np.concatenate([self.obs_preproc(obs[:-1]), acs], axis=-1))
            new_train_targs.append(self.targ_proc(obs[:-1], obs[1:]))
        self.train_in = np.concatenate([self.train_in] + new_train_in, axis=0).astype(np.float32)
        self.train_targs = np.concatenate([self.train_targs] + new_train_targs, axis=0).astype(np.float32)

        # Train the dynamics model
        self.model.fit_input_stats(self.train_in)
        self.model.reduce_params_among_models()

        idxs = np.random.randint(self.train_in.shape[0], size=[self.model.ensemble_size, self.train_in.shape[0]])

        num_batch = int(np.ceil(idxs.shape[-1] / self.batch_size))

        for epoch in range(self.epochs):
            for batch_num in range(num_batch):
                batch_idxs = idxs[:, batch_num * self.batch_size: (batch_num + 1) * self.batch_size]
                train_in = self.train_in[batch_idxs]
                train_targ = self.train_targs[batch_idxs]

                loss = self._update(_input=train_in, _target=train_targ)
        return loss

    def _update(self, _input, _target):
        with tf.GradientTape() as tape:
            loss = self._compute_loss(_input, _target)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def _compute_loss(self, _input, _target):
        loss = 0.01 * (tf.math.reduce_sum(self.model.max_logvar) - tf.math.reduce_sum(self.model.min_logvar))

        mean, logvar = self.model(_input, ret_logvar=True)
        self.model.reduce_L2_losses_among_models()

        inv_var = tf.math.exp(-logvar)

        train_losses = ((mean - _target) ** 2) * inv_var + logvar
        # TODO: why twice??
        train_losses = tf.math.reduce_mean(train_losses, axis=-1)
        train_losses = tf.math.reduce_mean(train_losses, axis=-1)
        train_losses = tf.math.reduce_sum(train_losses)
        # Only taking mean over the last 2 dimensions
        # The first dimension corresponds to each model in the ensemble

        loss += train_losses + self.model.decay_losses
        return loss

    def reset(self):
        """Resets this controller (clears previous solution, calls all update functions).

        Returns: None
        """
        self.prev_sol = np.tile((self.ac_lb + self.ac_ub) / 2, [self.plan_hor]).astype(np.float32)
        self.optimizer.reset()

        # for update_fn in self.update_fns:
        #     update_fn()

    @tf.contrib.eager.defun
    def act(self, obs, t, has_been_trained, get_pred_cost=False):
        # TODO: remove recursion... otherwise, this won't work... goes into an infinite loop...
        """Returns the action that this controller would take at time t given observation obs.

        Arguments:
            obs: The current observation
            t: The current timestep
            get_pred_cost: If True, returns the predicted cost for the action sequence found by
                the internal optimizer.

        Returns: An action (and possibly the predicted cost)
        """
        if not has_been_trained:
            return tf.random.uniform(shape=self.ac_lb.shape, minval=self.ac_lb, maxval=self.ac_ub)
        if self.ac_buf.shape[0] > 0:
            print("in")
            action, self.ac_buf = self.ac_buf[0], self.ac_buf[1:]
            return action

        self.sy_cur_obs = obs
        soln = self.optimizer.obtain_solution(self.prev_sol, self.init_var)
        self.prev_sol = tf.concat([soln[self.per * self.dU:], tf.zeros(self.per * self.dU)], axis=0)
        self.ac_buf = tf.reshape(soln[:self.per * self.dU], [-1, self.dU])

        # if self.ac_buf.shape[0] > 0:
        #     print("in")
        #     action, self.ac_buf = self.ac_buf[0], self.ac_buf[1:]
        #     return action
        return self.act(obs, t, has_been_trained)

    def _compile_cost(self, ac_seqs, get_pred_trajs=False):
        t, nopt = tf.constant(0), tf.shape(ac_seqs)[0]
        init_costs = tf.zeros([nopt, self.npart])
        ac_seqs = tf.reshape(ac_seqs, [-1, self.plan_hor, self.dU])
        ac_seqs = tf.reshape(tf.tile(
            tf.transpose(ac_seqs, [1, 0, 2])[:, :, None],
            [1, 1, self.npart, 1]
        ), [self.plan_hor, -1, self.dU])
        init_obs = tf.tile(self.sy_cur_obs[None], [nopt * self.npart, 1])

        def continue_prediction(t, *args):
            return tf.less(t, self.plan_hor)

        if get_pred_trajs:
            pred_trajs = init_obs[None]

            def iteration(t, total_cost, cur_obs, pred_trajs):
                cur_acs = ac_seqs[t]
                next_obs = self._predict_next_obs(cur_obs, cur_acs)
                delta_cost = tf.reshape(
                    self.obs_cost_fn(next_obs) + self.ac_cost_fn(cur_acs), [-1, self.npart]
                )
                next_obs = self.obs_postproc2(next_obs)
                pred_trajs = tf.concat([pred_trajs, next_obs[None]], axis=0)
                return t + 1, total_cost + delta_cost, next_obs, pred_trajs

            _, costs, _, pred_trajs = tf.while_loop(
                cond=continue_prediction, body=iteration, loop_vars=[t, init_costs, init_obs, pred_trajs],
                shape_invariants=[
                    t.get_shape(), init_costs.get_shape(), init_obs.get_shape(), tf.TensorShape([None, None, self.dO])
                ]
            )

            # Replace nan costs with very high cost
            costs = tf.reduce_mean(tf.where(tf.is_nan(costs), 1e6 * tf.ones_like(costs), costs), axis=1)
            pred_trajs = tf.reshape(pred_trajs, [self.plan_hor + 1, -1, self.npart, self.dO])
            return costs, pred_trajs
        else:
            def iteration(t, total_cost, cur_obs):
                cur_acs = ac_seqs[t]
                next_obs = self._predict_next_obs(cur_obs, cur_acs)
                delta_cost = tf.reshape(
                    self.obs_cost_fn(next_obs) + self.ac_cost_fn(cur_acs), [-1, self.npart]
                )
                return t + 1, total_cost + delta_cost, self.obs_postproc2(next_obs)

            _, costs, _ = tf.while_loop(
                cond=continue_prediction, body=iteration, loop_vars=[t, init_costs, init_obs]
            )

            # Replace nan costs with very high cost
            return tf.reduce_mean(tf.where(tf.is_nan(costs), 1e6 * tf.ones_like(costs), costs), axis=1)

    def _predict_next_obs(self, obs, acs):
        proc_obs = self.obs_preproc(obs)

        # TS Optimization: Expand so that particles are only passed through one of the networks.
        if self.prop_mode == "TS1":
            proc_obs = tf.reshape(proc_obs, [-1, self.npart, proc_obs.get_shape()[-1]])
            sort_idxs = tf.nn.top_k(
                tf.random_uniform([tf.shape(proc_obs)[0], self.npart]),
                k=self.npart
            ).indices
            tmp = tf.tile(tf.range(tf.shape(proc_obs)[0])[:, None], [1, self.npart])[:, :, None]
            idxs = tf.concat([tmp, sort_idxs[:, :, None]], axis=-1)
            proc_obs = tf.gather_nd(proc_obs, idxs)
            proc_obs = tf.reshape(proc_obs, [-1, proc_obs.get_shape()[-1]])
        if self.prop_mode == "TS1" or self.prop_mode == "TSinf":
            proc_obs, acs = self._expand_to_ts_format(proc_obs), self._expand_to_ts_format(acs)

        # Obtain model predictions
        inputs = tf.concat([proc_obs, acs], axis=-1)
        mean, var = self.model(inputs)
        predictions = mean

        # TODO: deal with this!!
        # if self.model.is_probabilistic and not self.ign_var:
        #     predictions = mean + tf.random_normal(shape=tf.shape(mean), mean=0, stddev=1) * tf.sqrt(var)
        #     if self.prop_mode == "MM":
        #         model_out_dim = predictions.get_shape()[-1].value
        #
        #         predictions = tf.reshape(predictions, [-1, self.npart, model_out_dim])
        #         prediction_mean = tf.reduce_mean(predictions, axis=1, keep_dims=True)
        #         prediction_var = tf.reduce_mean(tf.square(predictions - prediction_mean), axis=1, keep_dims=True)
        #         z = tf.random_normal(shape=tf.shape(predictions), mean=0, stddev=1)
        #         samples = prediction_mean + z * tf.sqrt(prediction_var)
        #         predictions = tf.reshape(samples, [-1, model_out_dim])
        # else:
        #     predictions = mean

        # TS Optimization: Remove additional dimension
        if self.prop_mode == "TS1" or self.prop_mode == "TSinf":
            predictions = self._flatten_to_matrix(predictions)
        if self.prop_mode == "TS1":
            predictions = tf.reshape(predictions, [-1, self.npart, predictions.get_shape()[-1]])
            sort_idxs = tf.nn.top_k(
                -sort_idxs,
                k=self.npart
            ).indices
            idxs = tf.concat([tmp, sort_idxs[:, :, None]], axis=-1)
            predictions = tf.gather_nd(predictions, idxs)
            predictions = tf.reshape(predictions, [-1, predictions.get_shape()[-1]])

        return self.obs_postproc(obs, predictions)

    def _expand_to_ts_format(self, mat):
        dim = mat.get_shape()[-1]
        return tf.reshape(
            tf.transpose(
                tf.reshape(mat, [-1, self.model.ensemble_size, self.npart // self.model.ensemble_size, dim]),
                [1, 0, 2, 3]
            ),
            [self.model.ensemble_size, -1, dim]
        )

    def _flatten_to_matrix(self, ts_fmt_arr):
        dim = ts_fmt_arr.get_shape()[-1]
        return tf.reshape(
            tf.transpose(
                tf.reshape(ts_fmt_arr, [self.model.ensemble_size, -1, self.npart // self.model.ensemble_size, dim]),
                [1, 0, 2, 3]
            ),
            [-1, dim]
        )