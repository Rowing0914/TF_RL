import gym
import environments.register as register
import numpy as np

# env_name = "CentipedeThree-v1"
env_name = "AntS-v1"
# env_name = "AntWithGoal-v1"
# env_name = "CentipedeFour-v1"
# env_name = "Humanoid-v2"
# env_name = "CentipedeFive-v1"

env = gym.make(env_name)
env.reset()
print(env.action_space.shape[0])

COEFF = 0.7

"""
Actuator Indices
[ankle_1, hip_2, ankle_2, hip_3, ankle_3, hip_4, ankle_4, hip_1]
[hip_1, ankle_1, hip_2, ankle_2, hip_3, ankle_3, hip_4, ankle_4]
"""

def leg1(env):
    leg_up = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ankle_move = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    pull_body = [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    actions = [leg_up, ankle_move, pull_body]

    for i in range(3):
        env.render()
        env.step(np.array(actions[i])*COEFF)
    return env

def leg2(env):
    leg_up = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ankle_move = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    pull_body = [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0]

    actions = [leg_up, ankle_move, pull_body]

    for i in range(3):
        env.render()
        env.step(np.array(actions[i])*COEFF)
    return env

def leg3(env):
    leg_up = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    ankle_move = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    pull_body = [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0]

    actions = [leg_up, ankle_move, pull_body]

    for i in range(3):
        env.render()
        env.step(np.array(actions[i])*COEFF)
    return env

def leg4(env):
    leg_up = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ankle_move = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    pull_body = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]

    actions = [leg_up, ankle_move, pull_body]

    for i in range(3):
        env.render()
        env.step(np.array(actions[i])*COEFF)
    return env

temp = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0]]

# while True:
#     for i in range(1000):
#         env.render()
#         env.step(temp[i%2])

while True:
    for i in range(1000):
        # env = eval("leg{}".format(1))(env)
        env = eval("leg{}".format((i%4)+1))(env)
