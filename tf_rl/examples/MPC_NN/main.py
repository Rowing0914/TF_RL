import gym
import numpy as np
import matplotlib.pyplot as plt

from tf_rl.common.eager_util import eager_setup
from tf_rl.examples.MPC_NN.utils.agents import RandomAgent, MPC
from tf_rl.examples.MPC_NN.utils.cost_fn import cheetah_cost_fn
from tf_rl.examples.MPC_NN.utils.model import DynamicsModel
from tf_rl.examples.MPC_NN.utils.utils import rollout, normalise

eager_setup()

env = gym.make("HalfCheetah-v2")
agent = RandomAgent(num_action=env.action_space.shape[0])
mpc = MPC(agent, cost_fn=cheetah_cost_fn)
model = DynamicsModel(state_shape=env.observation_space.shape[0])

losses = list()

# === Train the Dynamics Model ===
print("=== Start: Training Dynamics Model ===")
for epoch in range(10):
    states, actions, next_states, rewards, dones = rollout(env, agent, num_rollouts=3, horizon=500)
    states, actions, deltas, mean_delta, std_delta = normalise(states, actions, next_states)
    model.update_mean_std(mean_delta, std_delta)
    loss = model.update(states, actions, deltas)
    losses.append(loss.numpy())
print("=== Done: Training Dynamics Model ===")

# === Apply the MPC to the environment
print("=== Start: Applying MPC with Dynamics Model ===")
state = env.reset()
total_reward = 0
done = False
while not done:
    action = mpc.select_action(state, model)
    next_state, reward, done, info = env.step(action)
    total_reward += reward
print("Return of the episode: {}".format(total_reward))
print("=== Done: Applying MPC with Dynamics Model ===")

plt.plot(np.asarray(losses))
plt.grid()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Loss Curve")
plt.show()
