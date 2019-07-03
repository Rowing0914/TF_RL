import time, ray


# Execute f serially.

def f():
    time.sleep(1)
    return 1


print("==== Start ====")
results = [f() for i in range(4)]


# Execute f in parallel.

@ray.remote
def f():
    time.sleep(1)
    return 1


ray.init()
print("==== Start ====")
results = ray.get([f.remote() for i in range(4)])
print(results)
