import numpy as np
from tf_rl.common.random_process import OrnsteinUhlenbeckProcess

random_process = OrnsteinUhlenbeckProcess(size=1, theta=0.15, mu=1.0, sigma=0.2)

res = list()
for i in range(1000):
    res.append(random_process.sample())
    print(np.mean(res))