"""
In Hopper, as you can see from the source code in gym/hopper.py
they don't return the info item so that we cannot extract the useful information
from its returns of env.step.

Yet, the reward calculation is quite simple as follows:

```python
reward = (posafter - posbefore) / self.dt
reward += alive_bonus
reward -= 1e-3 * np.square(a).sum()
```

Hence, I didn't create any func to investigate the learned policy.

"""