import gym

env = gym.make("CartPole-v0")
state = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    state = next_state
env.close()
