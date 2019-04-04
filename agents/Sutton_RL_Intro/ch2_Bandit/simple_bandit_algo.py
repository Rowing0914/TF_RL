import numpy as np
import matplotlib.pyplot as plt

class Slot_Machine:
    def __init__(self, _min, _max):
        self._min = _min
        self._max = _max

    def slot_machine(self):
        return np.random.randint(low=self._min, high=self._max)

def decide_action(Q, epsilon):
    temp = np.random.rand(1,1)
    if temp >= (1-epsilon):
        return max(Q, key=Q.get)
    else:
        return str("sm{0}".format(np.random.randint(len(Q))))

def Bandit(A):
    return machines[A].slot_machine()

def logger(log_dict, target):
    result = []
    for i in target.items():
        result.append(i[1])
    log_dict.append(result)
    return log_dict

def softmax(vec):
    z = vec - max(vec)
    return np.exp(z)/np.sum(np.exp(z))

def log_policy(log_dict, target):
    result = []
    for i in target.items():
        result.append(i[1])
    tmp = softmax(np.array(result))
    log_dict.append(tmp)
    return log_dict

machines = dict()
ranges_for_slot_machine = [[1,3],
                           [3,5],
                           [1,15]]

# initialise slot machines
for index, row in enumerate(ranges_for_slot_machine):
    _min, _max = row
    machines['sm{0}'.format(index)] = Slot_Machine(_min, _max)

# initialise Q(a) and N(a)
Q = dict()
counter = dict()

for i in machines.items():
    Q[i[0]] = 0.3
    counter[i[0]] = 0

epochs = 10
epsilon = 0.1
learning_log_policy = []
learning_log_Q = []

for epoch in range(epochs):
    A = decide_action(Q, epsilon)
    R = Bandit(A)
    counter[A] += 1
    Q[A] += (1/counter[A])*(R - Q[A])
    learning_log_Q = logger(learning_log_Q, Q)
    learning_log_policy = log_policy(learning_log_policy, Q)

# vectorise the logs
learning_log_Q = np.array(learning_log_Q)
learning_log_policy = np.array(learning_log_policy)

# visualise the result
plt.subplot(211)
plt.plot(learning_log_Q)
plt.title("Action Value")
plt.subplot(212)
plt.plot(learning_log_policy)
plt.title("Policy")
plt.show()
