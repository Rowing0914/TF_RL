## Introduction

this is the repo for experimenting the RL algorithms



## Directory Architecture

- agents: algorithms
- common: utility functions
- test: testing the algorithms
- experiment: reproduce a result of a paper



## Algorithms

### Reinforcement Learning: R.Sutton's Great Book!

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

### Deep Reinforcement Learning

1. Playing Atari with Deep Reinforcement Learning, Mnih et al., 2013 [[arxiv]](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) [[code]](<https://github.com/Rowing0914/TF_RL/blob/master/agents/DQN_Atari.py>)
2. Deep Reinforcement Learning with Double Q-learning, van Hasselt et al., 2015 [[arxiv]](https://arxiv.org/abs/1509.06461) [[code]](<https://github.com/Rowing0914/TF_RL/blob/master/agents/Double_DQN_Atari.py>)
3. 



## References

- [@dennybritz's great repo](<https://github.com/dennybritz/reinforcement-learning>)
- [my research repo](<https://github.com/Rowing0914/Reinforcement_Learning>)
- [OpenAI Baselines](<https://github.com/openai/baselines>)
- [Keras-rl](<https://github.com/keras-rl/keras-rl>)