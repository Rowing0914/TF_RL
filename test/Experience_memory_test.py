import gym

from tf_rl.common.memory import PrioritizedReplayBuffer
from tf_rl.common.utils import AnnealingSchedule

env = gym.make("CartPole-v0")
# memory = ReplayBuffer(100)
memory = PrioritizedReplayBuffer(10000, alpha=0.6)
Beta = AnnealingSchedule(start=0.4, end=1.0, decay_steps=50)

print("Memory contains {0} timesteps".format(len(memory)))

for i in range(10):
    state = env.reset()
    for t in range(100):
        # env.render()
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)

        # memory format is: state, action, reward, next_state, done
        memory.add(state, action, reward, next_state, done)

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
        state = next_state

env.close()

print("Memory contains {0} timesteps".format(len(memory)))
state, action, reward, next_state, done, weights, indices = memory.sample(batch_size=10, beta=Beta.get_value(1))

