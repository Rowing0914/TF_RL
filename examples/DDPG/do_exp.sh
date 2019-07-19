#!/usr/bin/env bash

for i in 0.0, 0.3, 0.6, 0.9; do
/usr/local/bin/python3.6 /home/noio0925/Desktop/research/TF_RL/examples/DDPG/DDPG_eager.py --mu=$i
done