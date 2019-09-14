from tf_rl.common.random_process import OrnsteinUhlenbeckProcess
import matplotlib.pylab as plt
import numpy as np

num_sample = 1000*3

mu = [0.0, 0.3, 0.6, 0.9]

temp_ou, temp_ga = list(), list()

for i in range(len(mu)):
    OU = OrnsteinUhlenbeckProcess(size=1, theta=0.15, mu=mu[i], sigma=0.2)
    ou = [OU.sample()[0] for _ in range(num_sample)]
    plt.plot(ou, label="mu = {}".format(mu[i]))

plt.legend()
plt.title("Ornstein-Uhlenbeck Process")
plt.ylabel("Noise")
plt.xlabel("Samples")
plt.tight_layout()
plt.savefig("./image/ou_noise.eps", metadata="eps")