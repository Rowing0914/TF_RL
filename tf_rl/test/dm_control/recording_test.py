import dmc2gym

env = dmc2gym.make(domain_name='point_mass', task_name='easy', seed=1)

done = False
obs = env.reset()
while not done:
    env.render()
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
