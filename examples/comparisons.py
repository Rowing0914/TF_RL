import argparse
import os
# from common.visualise import plot_comparison_graph

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="CartPole", help="game env type")
args = parser.parse_args()

if args.mode == "CartPole":
	models = [
		"Q_learning",
		"REINFORCE",
		"DQN_eager",
		"Duelling_DQN_eager",
		"Double_DQN_eager",
		"DQN_PER_eager",
		"Duelling_Double_DQN_PER_eager",
		"DQfD_eager"
	]

	for model in models:
		os.system("python3.6 {}.py --num_episode 2000".format(model))

	# plot_comparison_graph(models)

elif args.mode == "Atari":
	models = [
		"DQN_eager",
		"Duelling_DQN_eager",
		"Double_DQN_eager",
		"DQN_PER_eager",
		"Duelling_Double_DQN_PER_eager",
		"DQfD_eager"
	]

	for model in models:
		os.system("python3.6 {}.py --mode Atari".format(model))

	# plot_comparison_graph(models)