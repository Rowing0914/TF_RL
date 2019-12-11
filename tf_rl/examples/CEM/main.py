import gym
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import deque


def evaluation(env, W, render=False):
    """ Go through 1 episode with a given collection of params """
    ep_reward = 0
    state = env.reset()
    done = False
    while not done:
        action = int(np.dot(W, state) > 0)  # simple linear policy to choose an action
        state, reward, done, info = env.step(action)
        ep_reward += reward
        if render: env.render()
    return ep_reward


def init_params(mu, sigma, num_population):
    """ take vector of mus, vector of sigmas, create matrix such that """
    num_features = mu.shape[0]
    W = np.zeros((num_population, num_features))
    for feat_index in range(num_features):
        W[:, feat_index] = np.random.normal(loc=mu[feat_index], scale=sigma[feat_index] + 1e-17, size=(num_population,))
    return W


def get_constant_noise(step):
    return np.max(5 - step / 10., 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default='CartPole-v0', type=str)
    # parser.add_argument("--env_name", default='CartPole-v1', type=str)
    parser.add_argument("--num_episodes", default=100, type=int)
    parser.add_argument("--num_population", default=40, type=int)
    parser.add_argument("--top_K", default=8, type=int)
    parser.add_argument("--goal", default=195, type=int)
    # parser.add_argument("--goal", default=450, type=int)
    parser.add_argument("--render", default=False, type=bool)
    params = parser.parse_args()

    env = gym.make(params.env_name)

    # Gaussian Distribution with means(mu) and std(sigma) for each feature
    mu_vec = np.random.uniform(size=env.observation_space.shape)
    sigma_vec = np.random.uniform(low=0.001, size=env.observation_space.shape)

    running_reward = deque(maxlen=10)
    all_rewards = list()

    for ep in range(params.num_episodes):
        # Init a weight matrix
        W = init_params(mu_vec, sigma_vec, params.num_population)
        reward_sums = list()

        for index in range(params.num_population):
            # sample an episode reward based on populated policies in Weight matrix
            # Note: each row in the matrix represents the individual policy
            reward_sums.append(evaluation(env, W[index, :], params.render))

        # rank individuals among the population
        reward_sums = np.array(reward_sums)
        rankings = np.argsort(reward_sums)

        # select the K best params which achieve good performance
        top_vectors = W[rankings, :]
        top_vectors = top_vectors[-params.top_K:, :]

        # update mu and sigma by averaging selected individuals in the population
        for feat_index in range(top_vectors.shape[1]):
            mu_vec[feat_index] = top_vectors[:, feat_index].mean()
            sigma_vec[feat_index] = np.abs(top_vectors[:, feat_index].std() + get_constant_noise(ep))

        running_reward.append(reward_sums.mean())
        all_rewards.append(reward_sums.mean())
        print("| Iter: {} | Mean R: {:.3f} | Moving Ave R(10ep): {:.3f} | Range of R: {} to {} |".format(
            ep, reward_sums.mean(), np.mean(running_reward), reward_sums.min(), reward_sums.max())
        )

        if np.mean(running_reward) > params.goal:
            break

    # Evaluate the trained policy's performance given the last episode
    score = evaluation(env, W[np.argmax(reward_sums), :], params.render)
    print("Final Score: {}".format(score))
    env.close()

    plt.plot(np.arange(len(all_rewards)), np.asarray(all_rewards), label="Reward Curve")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title("Score in {}".format(params.env_name))
    plt.grid()
    plt.legend()
    plt.savefig("./images/{}.png".format(params.env_name))
