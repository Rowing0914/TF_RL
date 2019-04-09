import os
from common.visualise import plot_comparison_graph

models = [
	"DQN",
	"Duelling_DQN",
	"Double_DQN",
	"DQN_PER",
	"Duelling_Double_DQN_PER",
]

for model in models:
	os.system("python3.6 ../agents/{}_train.py".format(model))

plot_comparison_graph(models)