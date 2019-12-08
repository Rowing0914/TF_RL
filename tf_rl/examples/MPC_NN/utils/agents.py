import numpy as np


class RandomAgent:
    def __init__(self, num_action):
        self.num_action = num_action

    def select_action(self, state):
        return np.random.randn(self.num_action)


class MPC:
    def __init__(self, agent, cost_fn, horizon=5, num_paths=10):
        self.agent = agent
        self.cost_fn = cost_fn
        self.horizon = horizon
        self.num_paths = num_paths

    def select_action(self, state, model):
        """ Given a state at t and the dynamics model, it traverses some trajectories within [horizon] time-steps,
            and finds the best trajectory minimising the cost.

            Note:
                this part is to do predict the future over some time-steps interacting with the dynamics model,
                you can change a lot of things in terms of
                - how to traverse a trajectory <= trajectory sampling so called
                - how to calculate(predict) rewards along with the trajectory <= inverse RL methods to predict reward
                - criterion to choose the best trajectory <= minimising a cost func is a good criterion??
                - and so on..
        """

        # TODO: there must be more concise and better design of algo....
        _state, _action_list, _next_state_list, = list(), list(), list()
        [_state.append(state) for _ in range(self.num_paths)]

        for _ in range(self.horizon):
            actions = [self.agent.select_action(_state[i]).tolist() for i in range(self.num_paths)]
            _state = model.predict(np.asarray(_state, dtype=np.float32), np.asarray(actions, dtype=np.float32))

            _action_list.append(actions)
            _next_state_list.append(_state)

        # Swap the axis from [horizon, num_paths, ...] => [num_paths, horizon, ...]
        actions = np.asarray(_action_list, dtype=np.float32).transpose(1, 0, 2)
        next_states = np.asarray(_next_state_list, dtype=np.float32).transpose(1, 0, 2)
        cost_per_traj = self.cost_fn(actions, next_states)
        traj_id = np.argmin(cost_per_traj)
        return np.asarray(actions)[traj_id, 0, :]  # take the first action in the trajectory minimising the cost
