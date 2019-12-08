import numpy as np


def rollout(env, agent, num_rollouts=10, horizon=1000):
    s, a, ns, r, d = list(), list(), list(), list(), list()
    for _ in range(num_rollouts):
        state = env.reset()
        done = False
        time_step = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            # each time-step
            s.append(state)
            a.append(action)
            ns.append(next_state)
            r.append(reward)
            d.append(done)
            state = next_state

            time_step += 1
            if time_step >= horizon:
                break
    return np.asarray(s, dtype=np.float32), np.asarray(a, dtype=np.float32), \
           np.asarray(ns, dtype=np.float32), np.asarray(r, dtype=np.float32), np.asarray(d, dtype=np.float32)


def normalise(states, actions, next_states):
    mean_s = np.mean(states, axis=0)
    mean_ns = np.mean(next_states, axis=0)
    mean_a = np.mean(actions, axis=0)
    mean_delta = np.mean(next_states - states, axis=0)

    std_s = np.std(states, axis=0)
    std_ns = np.std(next_states, axis=0)
    std_a = np.std(actions, axis=0)
    std_delta = np.std(next_states - states, axis=0)

    states = (states - mean_s) / (std_s + 1e-7)
    next_states = (next_states - mean_ns) / (std_ns + 1e-7)
    actions = (actions - mean_a) / (std_a + 1e-7)
    deltas = next_states - states
    return states, actions, deltas, mean_delta, std_delta
