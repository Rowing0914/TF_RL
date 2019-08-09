from tf_rl.common.random_process import GaussianNoise, OrnsteinUhlenbeckProcess
import matplotlib.pylab as plt

num_sample = 1000*3

for mu in [0.0, 0.3, 0.6, 0.9]:
    OU = OrnsteinUhlenbeckProcess(size=1, theta=0.15, mu=mu, sigma=0.2)
    ou = [OU.sample() for _ in range(num_sample)]
    Gaussian = GaussianNoise(mu=mu, sigma=0.2)
    gaussian = [Gaussian.sample() for _ in range(num_sample)]
    plt.plot(range(num_sample), ou, label="OU mu:{}".format(str(mu)))
    plt.plot(range(num_sample), gaussian, label="Gaussian mu:{}".format(str(mu)))

plt.title("Gaussian Noise vs Ornstein-Uhlenbeck Process")
plt.xlabel("Number of samples")
plt.ylabel("Value")
plt.legend()
plt.axes()
plt.grid()
plt.savefig("./image/gaussian_vs_ou_test.eps", metadata="eps")