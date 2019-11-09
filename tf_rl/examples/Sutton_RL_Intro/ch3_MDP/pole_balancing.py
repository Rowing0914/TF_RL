# reference: https://gym.openai.com/docs/
import gym

env = gym.make('CartPole-v0')

for episode in range(1000):
    observation = env.reset()
    for t in range(100):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
