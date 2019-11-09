# original code: https://gist.github.com/andrewliao11/d52125b52f76a4af73433e1cf8405a8f
import gym
import numpy as np
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
    env = gym.make('CartPole-v0')

    # Gaussian Distribution with means(mu) and std(sigma) for each feature
    mu_vec = np.random.uniform(size=env.observation_space.shape)
    sigma_vec = np.random.uniform(low=0.001, size=env.observation_space.shape)

    running_reward = deque(maxlen=10)
    num_episode = 30
    num_population = 40
    top_K = 8
    render = False

    for ep in range(num_episode):
        # Init a weight matrix
        W = init_params(mu_vec, sigma_vec, num_population)
        reward_sums = list()

        for index in range(num_population):
            # sample an episode reward based on populated policies in Weight matrix
            # Note: each row in the matrix represents the individual policy
            reward_sums.append(evaluation(env, W[index, :], render))

        # rank individuals among the population
        reward_sums = np.array(reward_sums)
        rankings = np.argsort(reward_sums)

        # select the K best params which achieve good performance
        top_vectors = W[rankings, :]
        top_vectors = top_vectors[-top_K:, :]

        # update mu and sigma by averaging selected individuals in the population
        for feat_index in range(top_vectors.shape[1]):
            mu_vec[feat_index] = top_vectors[:, feat_index].mean()
            sigma_vec[feat_index] = top_vectors[:, feat_index].std() + get_constant_noise(ep)

        running_reward.append(reward_sums.mean())
        print("| Iter: {} | Mean R: {:.3f} | Moving Ave R(10ep): {:.3f} | Range of R: {} to {} |".format(
            ep, reward_sums.mean(), np.mean(running_reward), reward_sums.min(), reward_sums.max())
        )

    # Evaluate the trained policy's performance given the last episode
    score = evaluation(env, W[np.argmax(reward_sums), :], render)
    print(score)
    env.close()
