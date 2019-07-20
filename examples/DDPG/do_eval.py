import os

for mu in ["00", "03", "06", "09"]:
    myCmd = "/usr/local/bin/python3.6 /home/noio0925/Desktop/research/TF_RL/examples/DDPG/eval.py --model_name=mu{}".format(mu)
    os.system(myCmd)