import numpy as np
from tf_rl.common.utils import copy_dir, delete_files
from tf_rl.common.abs_path import ROOT_DIR


def eval_Agent(agent, env, n_trial=1, google_colab=False):
    """ Evaluate the trained agent """
    all_rewards = list()
    for ep in range(n_trial):
        if ep == 0: env.record_start()
        state = np.array(env.reset())
        done = False
        episode_reward = 0
        while not done:
            # epsilon-greedy for evaluation using a fixed epsilon of 0.05(Nature does this!)
            action = agent.select_action_eval(state, epsilon=0.05)
            next_state, reward, done, _ = env.step(action)
            state = np.array(next_state)
            episode_reward += reward

        if ep == 0: env.record_end()

        all_rewards.append(episode_reward)
        print("| Evaluation | Ep: {}/{} | Score: {} |".format(ep + 1, n_trial, episode_reward))

    score = np.array([all_rewards]).mean()

    if google_colab:
        delete_files(folder="/content/gdrive/My Drive/TF_RL/logs/")
        copy_dir(src=ROOT_DIR+"/logs/", dst="/content/gdrive/My Drive/TF_RL/logs/")
    return score
