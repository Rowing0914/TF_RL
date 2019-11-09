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
                            file_name="exploration_eval_DDPG_{}.png".format(tf.compat.v1.train.get_global_step().numpy()))
        env.vis_trajectory(traj=traj, file_name="traj_eval_DDPG_{}.png".format(tf.compat.v1.train.get_global_step().numpy()))
        tf.contrib.summary.scalar("Evaluation Score", episode_reward, step=agent.index_timestep)
        print("| Ep: {}/{} | Score: {} |".format(ep + 1, n_trial, episode_reward))


def make_grid_env(plot_path):
    """ Create and instantiate a 2D Grid world """
    dense_goals = [(13.0, 8.0), (18.0, 11.0), (20.0, 15.0), (22.0, 19.0)]
    env = GridWorld(max_episode_len=500, num_rooms=1, action_limit_max=1.0, silent_mode=True,
                    start_position=(8.0, 8.0), goal_position=(22.0, 22.0), goal_reward=+100.0,
                    dense_goals=dense_goals, dense_reward=+5,
                    grid_len=30, plot_path=plot_path)
    return env


def visualise_critic_values(env, agent, flg="DDPG"):
    """ Visualise the Critic using a meshgrid to traverse the whole 2D world """
    high, low = env.observation_space.high, env.observation_space.low
    x = np.linspace(start=low[0], stop=high[0], num=1000)
    y = np.linspace(start=low[1], stop=high[1], num=1000)

    print("=== Evaluation Mode ===")
    data = np.stack([x, y], axis=-1).astype(np.float32)

    if flg == "DDPG":
        res = agent.critic(data, agent.eval_predict(data)).numpy().flatten()
    elif flg == "SAC":
        res = agent.critic(data, agent.eval_predict(data))[0].numpy().flatten()
    else:
        assert False, "Choose either DDPG or SAC"

    print(res.shape, x.shape, x.shape)
    print("Max Q: {}, Min Q: {}".format(res.max(), res.min()))

    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata

    X, Y = np.meshgrid(x, y)
    Z = griddata((x, y), res, (X, Y), method='nearest')
    env._visualise_setup()
    plt.contourf(X, Y, Z)
    plt.colorbar()
    plt.show()