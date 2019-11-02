import numpy as np
import tensorflow as tf


def eval_Agent(agent, env, n_trial=1):
    """ Evaluate the trained agent """
    all_rewards = list()
    for ep in range(n_trial):
        if ep == 0: env.record_start()
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            # epsilon-greedy for evaluation using a fixed epsilon of 0.05(Nature does this!)
            action = agent.select_action_eval(state, epsilon=0.05)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            episode_reward += reward

        if ep == 0: env.record_end()

        all_rewards.append(episode_reward)
        print("| Evaluation | Ep: {}/{} | Score: {} |".format(ep + 1, n_trial, episode_reward))

    rewards = np.array([all_rewards]).mean()
    tf.compat.v2.summary.scalar("eval/Score", rewards, step=tf.compat.v1.train.get_global_step())
