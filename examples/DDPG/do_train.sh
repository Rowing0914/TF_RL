#!/usr/bin/env bash

# this is the script which is supposed to execute bunch of experiments..
# but it's been weird, for example, when I execute this, I was expecting to see progress on console tho,
# it hasn't shown nothing on the console so that I've decided to use Python version of this.

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/noio0925/.mujoco/mujoco200/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-410

for i in 0.0 0.3 0.6 0.9; do
/usr/local/bin/python3.6 /home/noio0925/Desktop/research/TF_RL/examples/DDPG/DDPG_eager.py --mu=$i
done