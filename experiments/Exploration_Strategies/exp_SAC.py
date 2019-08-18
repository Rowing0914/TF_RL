import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

for seed in [10, 20, 30]:
    myCmd = "python3.6 ./SAC.py --seed={}".format(seed)
    os.system(myCmd)