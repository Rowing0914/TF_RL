import itertools, sys
import numpy as np
from tf_rl.env.continuous_gridworld.env import GridWorld

dense_goals = [(13.0, 8.0), (18.0, 11.0), (20.0, 15.0), (22.0, 19.0)]
env = GridWorld(max_episode_len=500, num_rooms=1, action_limit_max=1.0, silent_mode=True,
                start_position=(8.0, 8.0), goal_position=(22.0, 22.0), goal_reward=+100.0,
                dense_goals=dense_goals, dense_reward=+5,
                grid_len=30, plot_path="./logs/plots/RandomAgent/")

traj = list()
max_timestep = 500_000
global_timestep = 0
flag = False

for ep in itertools.count():
    state = env.reset()
    while True:
        global_timestep += 1
        sys.stdout.write("\r TimeStep: {0}".format(str(global_timestep)))
        sys.stdout.flush()
        traj.append(state)
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)

        # Same as the freq of eval phase in other algos
        if global_timestep % 100_000 == 0:
            flag = True

        if done:
            if flag:
                flag = False
                env.vis_exploration(traj=np.array(traj), file_name="exploration_{}.png".format(global_timestep))
                env.vis_trajectory(traj=np.array(traj), file_name="traj_{}.png".format(global_timestep))
            break
    if global_timestep > max_timestep:
        break