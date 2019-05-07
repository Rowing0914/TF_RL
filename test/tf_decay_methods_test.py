import matplotlib.pyplot as plt
import numpy as np

learning_rate = 1
end_learning_rate = 0.01
decay_steps = 100
decay_rate = 0.96

"""
Exponential Decay function

=== Computation Details ====
decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
"""


def exp_decay(global_step):
	return learning_rate * decay_rate ** (global_step / decay_steps)


"""
Polynomial Decay function

=== Computation Details ====
global_step = min(global_step, decay_steps)
decayed_learning_rate = (learning_rate - end_learning_rate) * (1 - global_step / decay_steps) ^ (power) + end_learning_rate
"""


def pol_decay(learning_rate, global_step, decay_steps, end_learning_rate, power=0.5):
	global_step = min(global_step, decay_steps)
	return (learning_rate - end_learning_rate) * (1 - global_step / decay_steps) ** (power) + end_learning_rate


"""
Inverse Time Decay function

=== Computation Details ====
decayed_learning_rate = learning_rate / (1 + decay_rate * floor(global_step / decay_step))
"""


def inv_decay(learning_rate, global_step, decay_steps, decay_rate):
	return learning_rate / (1 + decay_rate * np.floor(global_step / decay_steps))


# exp, pol, inv = list(), list(), list()
#
# for global_step in range(decay_steps+10):
# 	exp.append(exp_decay(starter_learning_rate, decay_rate, global_step, decay_steps))
# 	pol.append(pol_decay(starter_learning_rate,global_step, decay_steps, end_learning_rate, power=1.0))
# 	inv.append(inv_decay(starter_learning_rate,global_step,decay_steps,decay_rate))
#
# print(exp, pol, inv)

from tf_rl.common.utils import AnnealingSchedule


def func2(t):
	return np.exp(-t / 1.2)


a = AnnealingSchedule(start=0.01, end=1e-4, decay_steps=10000, decay_type="curved")
b = AnnealingSchedule(start=0.01, end=1e-4, decay_steps=10000, decay_type="linear")
temp, temp2 = list(), list()

for i in range(10000):
	temp.append(a.get_value(i))
	# temp.append(func2(i))
	# temp.append(0.01*0.96**i)
# temp2.append(b.get_value(i))
# temp2.append(exp_decay(global_step=i))

print(temp)
print(temp2)
plt.plot(temp, label="curved")
plt.plot(temp2, label="linear")
plt.legend()
plt.show()
