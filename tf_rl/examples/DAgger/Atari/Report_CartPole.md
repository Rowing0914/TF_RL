## Report ~DAgger~

### Algorithm

<img src="/home/noio0925/Desktop/CW01/images/DAgger_algo.png" width=60%>

### Method

1. We collect the dataset using the pre-trained agent(DQN)
2. We pre-trained the DAgger agent at this point
3. With the collected dataset, we trained the DAgger agent in online manner
4. We examined the trained DAgger agent on CartPole env and aggregated the rewards to see how well it was trained

### Parameters

- Game env: CartPole-v0 on OpenAI
- Number of episodes for pre-training DAgger agent: 100
- Number of action: 2 (move right or left)
- Epochs for training the neural network: 5
- Number of episodes for validation: 10
- Beta: 1, meaning in dataset collection we only rely on the DQN agent

#### Result

```shell
# result of the trained agent in 10 episodes
Score:  10.0
Score:  10.0
Score:  9.0
Score:  10.0
Score:  9.0
Score:  9.0
Score:  10.0
Score:  10.0
Score:  10.0
Score:  10.0

# result of the random agent in 10 episodes
Score: 15.0
Score: 27.0
Score: 17.0
Score: 18.0
Score: 18.0
Score: 25.0
Score: 25.0
Score: 12.0
Score: 16.0
Score: 33.0
```

### Analysis

- obviously the training does not affect well.
- maybe we need to increase the epochs for training the neural network
- I guess the task is too simple to apply the function approximation with neural network, but the thing is DQN agent works well.... so hmm..
- Principle cause could be the performance of DQN, since it works well in the dataset collection phase, we could only get the good examples which is normally the pole is located centre and looks like freezing. Hence, the neural network only learns from good examples and once it makes mistakes then instantly it encountered the new state in which the agent don't know how to behave.