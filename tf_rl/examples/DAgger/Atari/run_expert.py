import gym
import numpy as np
from gym.wrappers import Monitor
from load_policy import Duelling_DQN_Agent, DQN_Agent

DQN_Agent_model = "./expert_models/dqn_CartPole-v0_weights.h5f"
Duelling_DQN_Agent_model = "./expert_models/duel_dqn_CartPole-v0_weights.h5f"
OBS_FILE_PATH = "./expert_data/obs.npy"
ACT_FILE_PATH = "./expert_data/act.npy"

ENV_NAME     = "CartPole-v0"
OBSERVATION_SPACE = (4,) # for CartPole-v0
NB_ACTIONS = 2           # for CartPole-v0
AGENT_NAME   = "DQN" # In training, this outperforms over Duelling DQN for this simple CartPole task... curious..
NUM_ROLLOUTS = 100    # Number of expert roll outs
RENDER       = False  # if you want to see the expert's performance


def expert_play(agent, env):
	"""
	Let an experts play with the OpenAI game(default is CartPole-v0)

	Args:
		agent: the pre-trained agent, currently available agnets are below
			- DQN
			- Duelling DQN
	"""
	actions, observations = list(), list()

	for i_episode in range(NUM_ROLLOUTS):
		observation = env.reset()
		done = False
		while not done:
			action = agent.forward(observation)
			observation, reward, done, info = env.step(action)
			observations.append(observation)
			actions.append(action)
			if done:
				break

	return np.array(observations), np.array(actions)

def main():
	"""
	Main function
	"""
	if AGENT_NAME == "DQN":
		agent = DQN_Agent(DQN_Agent_model)
	elif AGENT_NAME == "DUELLING_DQN":
		agent = Duelling_DQN_Agent(Duelling_DQN_Agent_model)

	env = gym.make(ENV_NAME)
	env = Monitor(env, './videos', force=True)
	observations, actions = expert_play(agent, env)

	np.save("./expert_data/obs.npy", observations)
	np.save("./expert_data/act.npy", actions)

if __name__ == '__main__':
	main()
