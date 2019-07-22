import os

ENV_LIST = [
    # "Ant-v2",
    "HalfCheetah-v2",
    # "Hopper-v2",
    # "Humanoid-v2",
    # "Swimmer-v2",
    # "Walker2d-v2"
]

for env_name in ENV_LIST:
    for mu in ["00", "03", "06", "09"]:
        myCmd = "python3.6 /home/noio0925/Desktop/research/TF_RL/examples/DDPG/eval.py " \
                "--env_name={} --mu={}".format(env_name, mu)
        os.system(myCmd)