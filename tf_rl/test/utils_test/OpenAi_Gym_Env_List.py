"""
You can check the list of currently available envs.

Reference: https://stackoverflow.com/questions/48980368/list-all-environment-id-in-openai-gym/48989130#48989130

"""

from gym import envs

all_envs = envs.registry.all()

env_ids = [env_spec.id for env_spec in all_envs]

print("Currently {} Games are available".format(len(env_ids)))

# for env_id in env_ids:
#     print(env_id)
