# Intro
This directory for the experimental scripts.

- PPO: `ppos`
- DDPG: `DDPG_GGNN.py` is working but `DDPG_GCN.py` is not working.

# Usage
- ppo
```bash
cd ./ppos

# from https://github.com/reinforcement-learning-kr/pg_travel/blob/master/mujoco/main.py 
python ppo_main.py

# from https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO_continuous.py 
python ppo_second.py
```

- DDPG
```bash
python DDPG_GGNN.py
```