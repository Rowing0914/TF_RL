import gym

env = gym.make('CartPole-v0')
gym.wrappers.Monitor(env, './tmp/cartpole-experiment-1', force=True, video_callable=lambda episode_id: True)
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break

env.close()
