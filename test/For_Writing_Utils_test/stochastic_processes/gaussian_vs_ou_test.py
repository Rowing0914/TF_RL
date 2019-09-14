from tf_rl.common.random_process import GaussianNoise, OrnsteinUhlenbeckProcess
import matplotlib.pylab as plt
import numpy as np

num_sample = 1000*3

mu = [0.0, 0.3, 0.6, 0.9]
sigma = [0.1, 0.3, 0.6, 0.9]
ticks = ["0.0", "0.3", "0.6", "0.9"]

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

temp_ou, temp_ga = list(), list()

for i in range(len(mu)):
    OU = OrnsteinUhlenbeckProcess(size=1, theta=0.15, mu=mu[i], sigma=0.2)
    ou = [OU.sample()[0] for _ in range(num_sample)]
    Gaussian = GaussianNoise(mu=0.3, sigma=sigma[i])
    gaussian = [Gaussian.sample() for _ in range(num_sample)]
    temp_ou.append(ou)
    temp_ga.append(gaussian)


ou = plt.boxplot(temp_ou, positions=np.arange(len(mu))*2.0-0.4, sym='')
ga = plt.boxplot(temp_ga, positions=np.arange(len(mu))*2.0+0.4, sym='')

set_box_color(ou, "#D7191C") # colors are from http://colorbrewer2.org/
set_box_color(ga, "#2C7BB6")

# draw temporary red and blue lines and use them to create a legend
plt.plot([], c="#D7191C", label="OU Process")
plt.plot([], c="#2C7BB6", label="Gaussian")
plt.legend()

plt.title("Gaussian Noise vs Ornstein-Uhlenbeck Process")
plt.ylabel("Noise")
plt.xticks(range(0, len(ticks) * 2, 2), ticks)
plt.xlim(-2, len(ticks)*2)
plt.ylim(-3.0, 3.0)
plt.tight_layout()
plt.savefig("./image/gaussian_vs_ou_noise.eps", metadata="eps")