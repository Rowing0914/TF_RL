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
    # Use 5 seeds at least
    # https://arxiv.org/pdf/1806.08295.pdf
    for seed in [10, 20, 30, 40, 50]:
        myCmd = "python3.6 /home/noio0925/Desktop/research/TF_RL/examples/DDPG/DDPG_batch_norm.py " \
                "--env_name={} --seed={}".format(env_name, seed)
        os.system(myCmd)
