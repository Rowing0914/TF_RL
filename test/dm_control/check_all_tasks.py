from dm_control import suite

TASK_LIST = suite.ALL_TASKS

for data in TASK_LIST:
    domain, task = data
    print("Domain: {}, Task: {}".format(domain, task))