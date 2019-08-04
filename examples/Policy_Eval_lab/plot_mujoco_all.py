import os

ENV_LIST = [
    "Ant",
    "HalfCheetah",
    "Hopper",
    "Humanoid",
    "Swimmer",
    "Walker2d"
]

for env_name in ENV_LIST:
    myCmd = "python3.6 /home/noio0925/Desktop/research/TF_RL/examples/Policy_Eval_lab/MuJoCo/{}.py".format(env_name)
    os.system(myCmd)