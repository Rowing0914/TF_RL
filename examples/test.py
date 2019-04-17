import gym
from collections import deque
from tf_rl.common.wrappers import MyWrapper
from examples.params import Parameters, logdirs
from examples.DQN_eager import Model_CartPole
from tf_rl.common.memory import ReplayBuffer
from tf_rl.common.utils import AnnealingSchedule
from tf_rl.common.policy import EpsilonGreedyPolicy_eager
from tf_rl.agents.DQN import DQN
from tf_rl.common.train import test_Agent

logdirs = logdirs()

env = MyWrapper(gym.make("CartPole-v0"))
params = Parameters(mode="CartPole")
replay_buffer = ReplayBuffer(params.memory_size)
agent = DQN(Model_CartPole, Model_CartPole, env.action_space.n, params, logdirs.model_DQN)
Epsilon = AnnealingSchedule(start=params.epsilon_start, end=params.epsilon_end, decay_steps=params.decay_steps)
policy = EpsilonGreedyPolicy_eager(Epsilon_fn=Epsilon)
agent.check_point.restore(agent.manager.latest_checkpoint)
print("Restore model from disk")

test_Agent(agent, env, policy)