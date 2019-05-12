## TF-RL(Reinforcement Learning with Tensorflow: EAGER!!)

  This is the repo for implementing and experimenting the variety of RL algorithms using **Tensorflow Eager Execution**. And, since our Lord Google gracefully allows us to use their precious GPU resources without almost restriction, I have decided to enable most of codes run on **Google Colab**. So, if you don't have GPUs, please feel free to try it out on **Google Colab**



## Installation

- Install from Pypi(Test)

```shell
# this one
$ pip install --index-url https://test.pypi.org/simple/ --no-deps TF_RL
# or this one
$ pip install -i https://test.pypi.org/simple/ TF-RL
```

- Install from Github source

```shell
git clone https://github.com/Rowing0914/TF_RL.git
cd TF_RL
python setup.py install
```



## Features

1. Real time visualisation of an agent after training

<img src="assets/test_monitor.png" width="70%">

2. Comparison: Performance of algorithms using Tensorboard

<img src="assets/result.gif" width="70%">

```shell
$ cd examples
$ python3.6 comparisons.py
$ tensorboard --logdir=./logs/
```

3. Unit Test of a specific algorithm

```shell
$ cd examples
# Eager Execution mode
$ python3.6 examples/{model_name}/{model_name}_eager.py
# Graph mode Tensorflow: most of them are still under development...
$ python3.6 examples/unstable/{model_name}_train.py
```

4. Ready-to-run on Google colab

```shell
# you can run on google colab, but make sure that there some restriction on session
# 1. 90 minutes session reflesh
# 2. 12 Hours session reflesh
# Assuming you execute cmds below on Google Colab Jupyter Notebook
$ !git clone https://github.com/Rowing0914/TF_RL.git
$ pip install --index-url https://test.pypi.org/simple/ --no-deps TF_RL
$ %cd TF_RL
$ python3.6 examples/{model_name}/{model_name}_eager.py --mode Atari --env_name={env_name} --google_colab=True

# === Execute On Your Local ===
# My dirty workaroud to avoid breaking the connection to Colab is to execute below on local PC
$ watch -n 3600 python3.6 {your_filename}.py

""" Save this code to {your_filename}.py
import pyautogui
import time

# terminal -> chrome or whatever
pyautogui.hotkey("alt", "tab")
time.sleep(0.5)
# reflesh a page
pyautogui.hotkey("ctrl", "r")
time.sleep(1)
# say "YES" to a confirmation dialogue
pyautogui.hotkey("Enter")
time.sleep(1)
# next page
pyautogui.hotkey("ctrl", "tab")
# check all page reload properly
pyautogui.hotkey("ctrl", "tab")
time.sleep(1)
# switch back to terminal
pyautogui.hotkey("alt", "tab")
time.sleep(0.5)
"""
```



## Implementations (To be re-ordered soon)

1. Playing Atari with Deep Reinforcement Learning, Mnih et al., 2013 [[arxiv]](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) [[code]](<https://github.com/Rowing0914/TF_RL/blob/master/examples/DQN/DQN_eager.py>)
2. Deep Reinforcement Learning with Double Q-learning, van Hasselt et al., 2015 [[arxiv]](https://arxiv.org/abs/1509.06461) [[code]](<https://github.com/Rowing0914/TF_RL/blob/master/examples/Double_DQN/Double_DQN_eager.py>)
3. Duelling Network Architectures for Deep Reinforcement Learning, Wang et al., 2016 [[arxiv]](https://arxiv.org/abs/1511.06581) [[code]](<https://github.com/Rowing0914/TF_RL/blob/master/examples/Duelling_Double_DQN_PER_eager.py>)
4. Prioritised Experience Replay, T.Shaul et al., 2015 [[arxiv]](https://arxiv.org/abs/1511.05952) [[code]](<https://github.com/Rowing0914/TF_RL/blob/master/examples/DQN_PER/DQN_PER_eager.py>)
5. Asynchronous Methods for Deep Reinforcement Learning, Mnih et al., 2016 [[arxiv]](<https://arxiv.org/pdf/1602.01783.pdf>) [[code]](<https://github.com/Rowing0914/TF_RL/blob/master/agents/A3C.py>)
6. Deep Q-learning from Demonstrations, T.Hester et al., 2017 [[arxiv]](<https://arxiv.org/pdf/1704.03732.pdf>) [[code]](<https://github.com/Rowing0914/TF_RL/blob/master/examples/DQfD/DQfD_eager.py>)
7. Actor-Critic Algorithms, VR Konda and JN Tsitsiklis., 2000 NIPS [[arxiv]](<https://papers.nips.cc/paper/1786-actor-critic-algorithms.pdf>) [[code]](<https://github.com/Rowing0914/TF_RL/blob/master/examples/AC/Actor_Critic_eager.py>)
8. Policy Gradient Methods for Reinforcement Learning with Function Approximation., RS Sutton et al., 2000 NIPS [[arxiv]](<https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf>) [[code]](<https://github.com/Rowing0914/TF_RL/blob/master/examples/REINFORCE/REINFORCE_eager.py>)
9. Continuous Control with Deep Reinforcement Leaning, TP Lillicrap et al., 2015 [[arxiv]](<https://arxiv.org/pdf/1509.02971.pdf>) [[code]](<https://github.com/Rowing0914/TF_RL/blob/master/examples/DDPG/DDPG_eager.py>)
10. Deep Recurrent Q-Learning for Partially Observable MDPs, M Hausknecht., 2015 [[arxiv]](<https://arxiv.org/abs/1507.06527>) [[code]](https://github.com/Rowing0914/TF_RL/blob/master/examples/DRQN/DRQN_eager.py)
11. Hindsight Experience Replay, M.Andrychowicz et al., 2017 [[arxiv]](<https://arxiv.org/pdf/1707.01495.pdf>) [[code]](https://github.com/Rowing0914/TF_RL/blob/master/examples/HER/HER_eager.py)


#### Future dev

1. Noisy Networks for Exploration, M.Fortunato et al., 2017 [[arxiv]](https://arxiv.org/abs/1706.10295)
2. Soft Actor-Critic
3. Distributed DQN etc..



### Textbook implementations: R.Sutton's Great Book!

- Ch2: Simple Bandit: [[code]](<https://github.com/Rowing0914/TF_RL/blob/master/agents/Sutton_RL_Intro/ch2_Bandit/simple_bandit_algo.py>)
- Ch3: MDP sample using OpenAI Gym: [[code]](<https://github.com/Rowing0914/TF_RL/blob/master/agents/Sutton_RL_Intro/ch3_MDP/pole_balancing.py>)
- Ch4: Dynamic Programming
  - [policy_evaluation.py](https://github.com/Rowing0914/TF_RL/blob/master/agents/Sutton_RL_Intro/ch4_DP/policy_evaluation.py)
  - [policy_iteration.py](https://github.com/Rowing0914/TF_RL/blob/master/agents/Sutton_RL_Intro/ch4_DP/policy_iteration.py)
  - [value_iteration.py](https://github.com/Rowing0914/TF_RL/blob/master/agents/Sutton_RL_Intro/ch4_DP/value_iteration.py)
- Ch5: Monte Carlo Methods
  - [first_visi_MC_control_without_ES.py](https://github.com/Rowing0914/TF_RL/blob/master/agents/Sutton_RL_Intro/ch5_MC/first_visi_MC_control_without_ES.py)
  - [first_visit_MC.py](https://github.com/Rowing0914/TF_RL/blob/master/agents/Sutton_RL_Intro/ch5_MC/first_visit_MC.py)
  - [first_visit_MC_ES.py](https://github.com/Rowing0914/TF_RL/blob/master/agents/Sutton_RL_Intro/ch5_MC/first_visit_MC_ES.py)
  - [off_policy_MC.py](https://github.com/Rowing0914/TF_RL/blob/master/agents/Sutton_RL_Intro/ch5_MC/off_policy_MC.py)
- Ch6: Temporal Difference Learning
  - [double_q_learning.py](https://github.com/Rowing0914/TF_RL/blob/master/agents/Sutton_RL_Intro/ch6_TD/double_q_learning.py)
  - [expected_sarsa.py](https://github.com/Rowing0914/TF_RL/blob/master/agents/Sutton_RL_Intro/ch6_TD/expected_sarsa.py)
  - [one_step_TD.py](https://github.com/Rowing0914/TF_RL/blob/master/agents/Sutton_RL_Intro/ch6_TD/one_step_TD.py)
  - [q_learning.py](https://github.com/Rowing0914/TF_RL/blob/master/agents/Sutton_RL_Intro/ch6_TD/q_learning.py)
  - [sarsa.py](https://github.com/Rowing0914/TF_RL/blob/master/agents/Sutton_RL_Intro/ch6_TD/sarsa.py)
- Ch7: n_step_Bootstraping
  - [n_step_TD.py](https://github.com/Rowing0914/TF_RL/blob/master/agents/Sutton_RL_Intro/ch7_n_step_Bootstraping/n_step_TD.py)
  - [n_step_offpolicy_sarsa.py](https://github.com/Rowing0914/TF_RL/blob/master/agents/Sutton_RL_Intro/ch7_n_step_Bootstraping/n_step_offpolicy_sarsa.py)
  - [n_step_sarsa.py](https://github.com/Rowing0914/TF_RL/blob/master/agents/Sutton_RL_Intro/ch7_n_step_Bootstraping/n_step_sarsa.py)
- Ch13: Policy Gradient
  - [Actor_Critic_CliffWalk.py](https://github.com/Rowing0914/TF_RL/blob/master/agents/Sutton_RL_Intro/ch13_Policy_Gradient/Actor_Critic_CliffWalk.py)
  - [Actor_Critic_MountainCar.py](https://github.com/Rowing0914/TF_RL/blob/master/agents/Sutton_RL_Intro/ch13_Policy_Gradient/Actor_Critic_MountainCar.py)
  - [REINFORCE.py](https://github.com/Rowing0914/TF_RL/blob/master/agents/Sutton_RL_Intro/ch13_Policy_Gradient/REINFORCE.py)
  - [REINFORCE_keras.py](https://github.com/Rowing0914/TF_RL/blob/master/agents/Sutton_RL_Intro/ch13_Policy_Gradient/REINFORCE_keras.py)



## Game Envs

### Atari Envs

```python
from tf_rl.common.wrappers import wrap_deepmind, make_atari
from tf_rl.common.params import ENV_LIST_NATURE, ENV_LIST_NIPS


# for env_name in ENV_LIST_NIPS:
for env_name in ENV_LIST_NATURE:
    env = wrap_deepmind(make_atari(env_name))
    state = env.reset()
    for t in range(10):
        # env.render()
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        # print(reward, next_state)
        state = next_state
        if done:
            break
    print("{}: Episode finished after {} timesteps".format(env_name, t + 1))
    env.close()
```

### CartPole-Pixel(Obs: Raw Pixels in NumpyArray)

```python
import gym
from tf_rl.common.wrappers import CartPole_Pixel

env = CartPole_Pixel(gym.make('CartPole-v0'))
for ep in range(2):
	env.reset()
	for t in range(100):
		o, r, done, _ = env.step(env.action_space.sample())
		print(o.shape)
		if done:
			break
env.close()
```



## PC Envs

- OS: Linux Ubuntu LTS 16.04
- Python: 3.6
- GPU: Gefoce GTX1060
- Tensorflow: 1.13.0
- CUDA: 10.0
- libcudnn: 7.4.1



### GPU Env Maintenance on Ubuntu 16.04 (CUDA 10)

  Check this link as well: <https://www.tensorflow.org/install/gpu>

```shell
# Add NVIDIA package repositories
# Add HTTPS support for apt-key
sudo apt-get install gnupg-curl
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_10.0.130-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604_10.0.130-1_amd64.deb
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
sudo apt-get update
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
sudo apt install ./nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
sudo apt-get update

# Install NVIDIA Driver
# Issue with driver install requires creating /usr/lib/nvidia
sudo mkdir /usr/lib/nvidia
sudo apt-get install --no-install-recommends nvidia-410
# Reboot. Check that GPUs are visible using the command: nvidia-smi

# Install development and runtime libraries (~4GB)
sudo apt-get install --no-install-recommends \
    cuda-10-0 \
    libcudnn7=7.4.1.5-1+cuda10.0  \
    libcudnn7-dev=7.4.1.5-1+cuda10.0


# Install TensorRT. Requires that libcudnn7 is installed above.
sudo apt-get update && \
        sudo apt-get install nvinfer-runtime-trt-repo-ubuntu1604-5.0.2-ga-cuda10.0 \
        && sudo apt-get update \
        && sudo apt-get install -y --no-install-recommends libnvinfer-dev=5.0.2-1+cuda10.0
```



## Disclaimer: you will see similar codes in different source codes.

  In this repo, I would like to ignore the efficiency in development, because although I have seen a lot of clean and neat implementations of DRL algorithms on the net, I think sometimes they excessively modularise some components by introducing a lot of extra parameters or flags which are not in the original papers, in other words, they are toooo professional for me to play with. 

  So, in this repo I do not hesitate to re-use the same codes here or there. **BUT** I believe this way of organising the algorithms enhances our understanding more compared to try making the algorithms compact.



## References

- if you get stuck at DQN, you may want to refer to this great guy's entry: <https://adgefficiency.com/dqn-debugging/>
- [@dennybritz's great repo](<https://github.com/dennybritz/reinforcement-learning>)
- [my research repo](<https://github.com/Rowing0914/Reinforcement_Learning>)
- [OpenAI Baselines](<https://github.com/openai/baselines>)
- [Keras-rl](<https://github.com/keras-rl/keras-rl>)