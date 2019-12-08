import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

ENV_LIST = [
    "AntWithGoal-v1",
    "CentipedeFour-v1"
]

for env_name in ENV_LIST:
    # Use 5 seeds at least
    # https://arxiv.org/pdf/1806.08295.pdf
    for seed in [10, 20, 30]:
        myCmd = "python3.6 ./ppo/ppo_cpu.py --env_name={} --seed={}".format(env_name, seed)
        os.system(myCmd)
