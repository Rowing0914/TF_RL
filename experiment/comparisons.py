import os
import numpy as np
import matplotlib.pyplot as plt

models = [
	"DQN_train",
	"Double_DQN_train",
	"DQN_PER_train",
	"Duelling_Double_DQN_PER_train",
]

# refresh the folder
os.system("rm ../logs/values/*")
os.system("rm ../logs/graphs/*")

for model in models:
	os.system("python3.6 ../agents/{}.py".format(model))

	reward = np.load("../logs/values/"+model+"_reward.npy")
	loss = np.load("../logs/values/"+model+"_loss.npy")

	# temporal visualisation
	plt.subplot(2, 1, 1)
	plt.plot(reward, label=model)
	plt.title("Score over time")
	plt.xlabel("Timestep")
	plt.ylabel("Score")

	plt.subplot(2, 1, 2)
	plt.plot(loss, label=model)
	plt.title("Loss over time")
	plt.xlabel("Timestep")
	plt.ylabel("Loss")

plt.legend(loc='upper right')
plt.savefig("../logs/graphs/comparison.png")
plt.show()