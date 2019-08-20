import numpy as np
import tensorflow as tf

from tf_rl.env.continuous_gridworld.env import GridWorld


def eval_Agent(env, agent, n_trial=1):
    """ Evaluate the trained agent with the recording of its behaviour """

    print("=== Evaluation Mode ===")
    traj = list()
    for ep in range(n_trial):
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            traj.append(state)
            action = agent.eval_predict(state)
            next_state, reward, done, info = env.step(np.clip(action, -1.0, 1.0))
            state = next_state
            episode_reward += reward

        traj = np.array(traj)
        env.vis_exploration(traj=traj,
                            file_name="exploration_eval_{}.png".format(tf.compat.v1.train.get_global_step().numpy()))
        env.vis_trajectory(traj=traj, file_name="traj_eval_{}.png".format(tf.compat.v1.train.get_global_step().numpy()))
        tf.contrib.summary.scalar("Evaluation Score", episode_reward, step=agent.index_timestep)
        print("| Ep: {}/{} | Score: {} |".format(ep + 1, n_trial, episode_reward))


def make_grid_env(plot_path):
    dense_goals = [(13.0, 8.0), (18.0, 11.0), (20.0, 15.0), (22.0, 19.0)]
    env = GridWorld(max_episode_len=500, num_rooms=1, action_limit_max=1.0, silent_mode=True,
                    start_position=(8.0, 8.0), goal_position=(22.0, 22.0), goal_reward=+100.0,
                    dense_goals=dense_goals, dense_reward=+5,
                    grid_len=30, plot_path=plot_path)
    return env
