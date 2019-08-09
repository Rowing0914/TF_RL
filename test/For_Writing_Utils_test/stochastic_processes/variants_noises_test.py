from stochastic.noise import *
import matplotlib.pylab as plt

PRCS = [
    BlueNoise,
    BrownianNoise,
    ColoredNoise,
    PinkNoise,
    RedNoise,
    VioletNoise,
    WhiteNoise,
    FractionalGaussianNoise,
    GaussianNoise
]

NUM_SAMPLE = 50

for PRC in PRCS:
    prc = PRC()
    print("=== {} ===".format(type(prc).__name__))
    s = prc.sample(NUM_SAMPLE)
    plt.plot(range(len(s)), s, label="{}".format(type(prc).__name__))

plt.legend()
plt.show()
