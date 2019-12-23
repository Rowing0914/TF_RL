"""
Algorithm: Dyna Q
Description: https://i.stack.imgur.com/Jqidx.png
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt

from tf_rl.examples.DynaQ.model import Model
from tf_rl.examples.DynaQ.agent import QAgent
from tf_rl.examples.Sutton_RL_Intro.libs.envs.grid_world import GridworldEnv


def train(env, agent, model, num_episode=100, num_train=5, hot_start=100):
    global_ts = 1
    all_rewards = list()

    # Collect samples, Update the dynamics table
    for episode in range(num_episode):
        state = env.reset()
        episode_reward = 0
        for t in itertools.count():
            if global_ts > hot_start:
                action = agent.select_action(state=state, epsilon=min(0.99 * global_ts, 0.01))  # temp annealing
            else:
                action = agent.select_action(state=state, epsilon=1.0)

            next_state, reward, done, info = env.step(action)
            episode_reward += reward

            # update the agent/dynamics model
            model.update(state=state, action=action, reward=reward, next_state=next_state)
            agent.update(state, action, reward, next_state, alpha=min(0.99 * global_ts, 0.01))  # temp annealing

            # update the Q Agent using the learnt dynamics model
            for epoch in range(num_train):
                state, action = model.sample()
                next_state, reward = model.step(state=state, action=action)
                agent.update(state, action, reward, next_state, alpha=min(0.99 * global_ts, 0.01))  # temp annealing

            if done:
                all_rewards.append(episode_reward)
                print("| Ep: {} | Reward: {}".format(episode, episode_reward))
                break
            else:
                state = next_state
                global_ts += 1

    env.close()
    return all_rewards


def train_eval(num_episode=100, num_train=5, hot_start=100):
    env = GridworldEnv()
    agent = QAgent(num_state=env.observation_space.n, num_action=env.action_space.n)
    model = Model(num_state=env.observation_space.n, num_action=env.action_space.n)
    all_rewards = train(env=env,
                        agent=agent,
                        model=model,
                        num_episode=num_episode,
                        num_train=num_train,
                        hot_start=hot_start)
    print("[Last 10 Ep] Mean Reward: {:.3f}".format((sum(all_rewards[-10:])*1.0) / 10.0))

    # quick visualisation
    plt.plot(np.asarray(all_rewards))
    plt.title("[DynaQ] Episodic Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.grid()
    plt.axis()
    plt.savefig("result.png")
    plt.clf()


def main(num_episode=100, num_train=5, hot_start=100):
    train_eval(num_episode=num_episode,
               num_train=num_train,
               hot_start=hot_start)


if __name__ == '__main__':
    num_episode = 100*3
    num_train = 10
    hot_start = 100
    main(num_episode=num_episode,
         num_train=num_train,
         hot_start=hot_start)
