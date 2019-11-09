import os

ENV_LIST = [
    "Ant-v2",
    "HalfCheetah-v2",
    "Hopper-v2",
    "Humanoid-v2",
    "Swimmer-v2",
    "Walker2d-v2"
]

for env_name in ENV_LIST:
    myCmd = "python3.6 /home/norio0925/Desktop/TF_RL/examples/DDPG/DDPG_all_logs.py --env_name={}".format(env_name)
    os.system(myCmd)
