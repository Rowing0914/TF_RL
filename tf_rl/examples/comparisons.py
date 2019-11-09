import argparse
import os

# from common.visualise import plot_comparison_graph

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="cartpole", help="game env type")
args = parser.parse_args()

if args.mode == "cartpole":
    models = [
        "DQN/DQN_eager",
        "Duelling_DQN/Duelling_DQN_eager",
        "Double_DQN/Double_DQN_eager",
        "DQN_PER/DQN_PER_eager",
        "DDDP/Duelling_Double_DQN_PER_eager",
        # "DQfD/DQfD_eager"
    ]

    for model in models:
        os.system(
            "python3.6 {0}_{1}.py --mode=CartPole --log_dir=../logs/logs/{0}/ --model_dir=../logs/models/{0}/".format(
                model, args.mode))

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
