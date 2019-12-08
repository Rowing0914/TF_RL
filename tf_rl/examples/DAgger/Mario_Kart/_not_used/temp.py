# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("--record", help="Record the demo and store the outcome to `temp` directory")
# parser.add_argument("--prepare", help="Get `data,csv` and screenshots of the game, then prepare features and labels in `data` directory")
# args = parser.parse_args()
# print(args)
# print(args.record)

# def a():
# 	while True:
# 		pass
# 		# print('hello')

# def b():
# 	print('hi')

# for i in range(3):
# 	print(i)
# 	try:
# 		a()
# 	except KeyboardInterrupt:
# 		b()

# import os

# for i in range(3):
# 	os.system("mkdir f_{0}".format(i))

import numpy as np
for i in range(2):
	if i == 0:
		base = np.load("data/X_ep_{0}.npy".format(i))
	else:
		temp = np.load("data/X_ep_{0}.npy".format(i))
		np.concatenate((base, temp), axis=0)
