## OpenAI Gym

- Link: <https://gym.openai.com/>

- Original Paper: Bellemare, Marc G., et al. "The arcade learning environment: An evaluation platform for general agents." *Journal of Artificial Intelligence Research* 47 (2013): 253-279.

- Num_Games: As of 4/4/2019, 797 Games are available

```python
from gym import envs

all_envs = envs.registry.all()
env_ids = [env_spec.id for env_spec in all_envs]
print("Currently {} Games are available".format(len(env_ids)))
```



## Naming rule of Environments

In Atari ALE, OpenAI follows some rules below

- {}-v0
- {}-v4
- {}Deterministic-v0
- {}Deterministic-v4
- {}NoFrameskip-v0
- {}NoFrameskip-v4
- {}-ram-v0
- {}-ram-v4
- {}-ramDeterministic-v0
- {}-ramDeterministic-v4
- {}-ramNoFrameskip-v0
- {}-ramNoFrameskip-v4



## Details

### v0 vs v4??

- v0: `repeat_action_probability = 0.25`
  - Gym repeats the same action at the previous time step with the probability of 0.25
- v0: `repeat_action_probability = 0.0`
  - Gym repeats the same action at the previous time step with the probability of 0.0

### ram or not??

- ram: Gym observes the RAM of the ALE environment and returns as an `obsevation`
- others: Frame of the ALE environment is an `observation`

Simply it is better **not ** to use ram option

### Deterministic/NoFrameskip or not

- Deterministic: Gym always repeat the action four times
- NoFrameskip: Gym does not repeat the action
- others: Gym randomly repeat the action between 2 to 4 times



## Tips

- If you want to manually repeat the same action, then **DO NOT** use *NoFrameskip*
- If you want to just focus on the algorithm without handling the repetition of actions, then use *Deterministic*