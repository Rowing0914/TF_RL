import gym, numpy as np, collections

def Momentum(action, past_actions, T=5, gamma=0.9, mode="normal"):
    # T represents the time-horizon
    if mode == "normal":
        # a = gamma * 1/T * sum(a_t-1) + a_t
        return gamma*np.mean(past_actions, axis=0) + action
    elif mode == "weighted":
        # a = 1/T (sum^T_k (gamma**(T-k)) * a_T-k) + a_t
        _inner = np.mean([gamma**(T-k)*past_actions[-k, :] for k in range(T)], axis=0)
        return _inner + action

env = gym.make("Ant-v2")
env.reset()
done = False
distance, actions = list(), collections.deque(maxlen=5)
while not done:
    env.render()
    action = env.action_space.sample()
    actions.append(action)
    print(action)
    action = Momentum(action, past_actions=np.array(actions), T=len(actions), mode="weighted")
    print(action)
    print()
    s, r, done, info = env.step(action)
env.close()