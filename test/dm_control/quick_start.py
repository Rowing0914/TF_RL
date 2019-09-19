from dm_control import suite
from dm_control.suite.wrappers import pixels
import numpy as np

DOMAIN_NAME = "acrobot"
TASK_NAME = "swingup"

# Load one task:
env = suite.load(domain_name=DOMAIN_NAME, task_name=TASK_NAME)

# Wrap the environment to obtain the pixels
env = pixels.Wrapper(env, pixels_only=False)

# Step through an episode and print out reward, discount and observation.
action_spec = env.action_spec()
time_step = env.reset()

while not time_step.last():
    action = np.random.uniform(action_spec.minimum,
                               action_spec.maximum,
                               size=action_spec.shape)
    time_step = env.step(action)
    observation_dm = time_step.observation["pixels"]
    print(observation_dm)
