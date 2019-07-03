import gym

env = gym.make("MountainCarContinuous-v0")
# env = gym.make("MountainCar-v0")

for i in range(5):
    state = env.reset()
    for t in range(100):
        env.render()
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        print(action, state, next_state, reward, done)
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
        state = next_state

env.close()
