import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

ENV_LIST = [
    "Ant-v2",
    "HalfCheetah-v2",
    "Hopper-v2",
    "Humanoid-v2",
    "Swimmer-v2",
    "Walker2d-v2"
]

for env_name in ENV_LIST:
    myCmd = "python3.6 /home/noio0925/Desktop/research/TF_RL/examples/DDPG/original_DDPG.py --env_name={}".format(env_name)
    os.system(myCmd)