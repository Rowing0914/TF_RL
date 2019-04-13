from tf_rl.common.utils import AnnealingSchedule
from tf_rl.common.params import Parameters
from tf_rl.common import EpsilonGreedyPolicy

params = Parameters("CartPole")
Epsilon = AnnealingSchedule(start=params.epsilon_start, end=params.epsilon_end, decay_steps=params.decay_steps)
policy = EpsilonGreedyPolicy(Epsilon_fn=Epsilon)
num_episodes = 80

for ep in range(num_episodes):
	print(Epsilon.get_value(ep))
	policy.index_episode = ep
	print(policy.current_epsilon())