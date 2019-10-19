import matplotlib.pyplot as plt
import numpy as np

a = np.random.rand(1000)

plt.hist(a, bins=100, alpha=0.3, density=True)
plt.xlabel("Distance")
plt.ylabel("Density")
plt.title("Histogram of Distance:")
# plt.axes()
plt.grid()
# plt.legend()
plt.show()