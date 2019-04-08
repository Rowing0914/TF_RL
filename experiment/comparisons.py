import os

models = [
	"DQN_train",
	"Duelling_DQN_train",
	"Double_DQN_train",
	"DQN_PER_train",
	"Duelling_Double_DQN_PER_train",
]

for model in models:
	os.system("python3.6 ../agents/{}.py".format(model))