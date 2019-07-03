import gym, ray, os, time

batch_size = 5
iteration = 100


@ray.remote
class PongEnv(object):
    def __init__(self):
        # Tell numpy to only use one core. If we don't do this, each actor may try
        # to use all of the cores and the resulting contention may result in no
        # speedup over the serial version. Note that if numpy is using OpenBLAS,
        # then you need to set OPENBLAS_NUM_THREADS=1, and you probably need to do
        # it from the command line (so it happens before numpy is imported).
        os.environ["MKL_NUM_THREADS"] = "1"
        self.env = gym.make("CartPole-v0")
        self.env.reset()

    def step(self, action):
        return self.env.step(action)


class _Env:
    def __init__(self, env):
        self.env = env

    def step(self, action):
        return self.env.step.remote(action)


ray.init(num_cpus=5)

actors = [_Env(env=PongEnv.remote()) for _ in range(batch_size)]

actions = []
start = time.time()
for i in range(int(iteration / batch_size)):
    for t in range(batch_size):
        action_id = actors[t].step(1)
        actions.append(action_id)
print("done: {}s".format(time.time() - start))
data = ray.get(actions)
print(type(data))
print(len(data))

result = []
env = gym.make("CartPole-v0")
env.reset()
start = time.time()
for i in range(iteration):
    temp = env.step(1)
    result.append(temp)
print("done: {}s".format(time.time() - start))
