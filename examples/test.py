import gym
from tf_rl.common.wrappers import MyWrapper
from tf_rl.common.params import Parameters, logdirs
from examples.DQN_eager import Model_CartPole
from tf_rl.common.policy import TestPolicy
from tf_rl.agents.DQN import DQN
from tf_rl.common.train import test_Agent

logdirs = logdirs()

env = MyWrapper(gym.make("CartPole-v0"))
params = Parameters(mode="CartPole")
agent = DQN(Model_CartPole, Model_CartPole, env.action_space.n, params, logdirs.model_DQN)
policy = TestPolicy()
agent.check_point.restore(agent.manager.latest_checkpoint)
print("Restore model from disk")

test_Agent(agent, env, policy)