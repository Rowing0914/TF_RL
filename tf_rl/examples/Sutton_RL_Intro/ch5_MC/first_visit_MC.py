# 5.1 Monte Carlo Prediction
# First-visit MC prediction, for estimating state-value

from collections import defaultdict
import sys

if "../" not in sys.path:
    sys.path.append("../")

from utils.envs.blackjack import BlackjackEnv


def First_Visit_MC(env, state_value, policy, discount_factor=1.0, num_episodes=1000):
    Returns = defaultdict(float)
    Returns_count = defaultdict(float)

    for i in range(num_episodes):
        # observe the environment and store the observation
        experience = []
        observation = env.reset()
        for t in range(100):
            action = policy(observation)
            next_observation, reward, done, _ = env.step(action)
            experience.append((observation, action, reward))
            observation = next_observation
            if done:
                break

        # remove duplicated states in the episode
        states_in_experience = set([(x[0]) for x in experience])
        # update the state-value function using the obtained episode
        for state in states_in_experience:
            # Find the first occurance of the state in the episode
            first_occurence_idx = next(i for i, x in enumerate(experience) if x[0] == state)
            # Sum up all discounted rewards over time since the first occurance in the episode
            G = sum([x[2] * (discount_factor ** i) for i, x in enumerate(experience[first_occurence_idx:])])
            # Calculate average return for this state over all sampled experiences
            Returns[state] += G
            Returns_count[state] += 1.0
            state_value[state] = Returns[state] / Returns_count[state]
    return state_value


def policy(observation):
    # Action => {0: "Stick", 1: "Hit"}
    return 0 if observation[0] >= 20 else 1


if __name__ == '__main__':
    env = BlackjackEnv()
    state_value = defaultdict(float)
    discount_factor = 1.0
    num_episodes = 10
    state_value = First_Visit_MC(env, state_value, policy, discount_factor=1.0, num_episodes=num_episodes)
    print(state_value)
