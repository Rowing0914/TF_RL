import numpy as np
from tf_rl.env.continuous_gridworld.env import GridWorld

dense_goals = [(13.0, 8.0), (18.0, 11.0), (20.0, 15.0), (22.0, 19.0)]
env = GridWorld(max_episode_len=500, num_rooms=1, action_limit_max=1.0, silent_mode=True,
                start_position=(8.0, 8.0), goal_position=(22.0, 22.0), goal_reward=+100.0,
                dense_goals=dense_goals, dense_reward=+5,
                grid_len=30, plot_path="./images")

state = env.reset()
traj = list()

for ep in range(3):
    for i in range(1000):
        action = env.action_space.sample()
        traj.append(state)
        state, reward, done, info = env.step(action)
        if done:
            state = env.reset()

traj = np.array(traj)
env.vis_exploration(traj=traj, file_name="exploration.png")
env.vis_trajectory(traj=traj, file_name="traj.png")