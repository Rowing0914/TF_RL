from stochastic.diffusion import *
import matplotlib.pylab as plt

PRCS = [
    ConstantElasticityVarianceProcess,
    CoxIngersollRossProcess,
    OrnsteinUhlenbeckProcess,
    VasicekProcess
]

NUM_SAMPLE = 1000

for PRC in PRCS:
    prc = PRC()
    print("=== {} ===".format(type(prc).__name__))
    s = prc.sample(NUM_SAMPLE)
    plt.plot(range(len(s)), s, label="{}".format(type(prc).__name__))

plt.legend()
plt.show()
