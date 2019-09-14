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
    for name in ["ou", "gaussian"]:
        # Use 5 seeds at least
        # https://arxiv.org/pdf/1806.08295.pdf
        # for seed in [10, 20, 30]:
        for seed in [10, 20]:
            myCmd = "python3.6 /home/norio0925/Desktop/TF_RL/examples/DDPG/DDPG_eager.py " \
                    "--env_name={} --random_process={} --seed={} --train_flg=original".format(env_name, name, seed)
            os.system(myCmd)
