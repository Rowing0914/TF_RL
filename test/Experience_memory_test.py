import gym

from common.memory import ReplayBuffer

env = gym.make("CartPole-v0")
memory = ReplayBuffer(100)

print("Memory contains {0} timesteps".format(len(memory)))

for i in range(10):
    state = env.reset()
    for t in range(100):
        env.render()
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)

        # memory format is: state, action, reward, next_state, done
        memory.store(state, action, reward, next_state, done)

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
        state = next_state

env.close()

print("Memory contains {0} timesteps".format(len(memory)))