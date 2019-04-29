import gym

from tf_rl.common.memory import PrioritizedReplayBuffer, ReplayBuffer
from tf_rl.common.utils import AnnealingSchedule

env = gym.make("CartPole-v0")
memory = ReplayBuffer(100)
# memory = PrioritizedReplayBuffer(10000, alpha=0.6)
# Beta = AnnealingSchedule(start=0.4, end=1.0, decay_steps=50)

print("Memory contains {0} timesteps".format(len(memory)))

for i in range(10):
    state = env.reset()
    episode_memory = list()
    for t in range(100):
        # env.render()
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)

        # memory format is: state, action, reward, next_state, done
        episode_memory.append((state, action, reward, next_state, done))

        if done:
            print("Episode finished after {} timesteps".format(t+1))

            s1, a, r, s2, d = [],[],[],[],[]
            for data in episode_memory:
                s1.append(data[0])
                a.append(data[1])
                r.append(data[2])
                s2.append(data[3])
                d.append(data[4])
            memory.add(s1, a, r, s2, d)
            break
        state = next_state

env.close()

print("Memory contains {0} timesteps".format(len(memory)))
# state, action, reward, next_state, done, weights, indices = memory.sample(batch_size=10, beta=Beta.get_value(1))
print(memory.sample(1))
