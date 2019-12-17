import os

PROJECT_NAME = "TF_RL"

_path = os.path.abspath(os.curdir)
ROOT_DIR = _path.split(PROJECT_NAME)[0] + PROJECT_NAME
ROOT_colab = "/content/gdrive/My Drive/{}".format(PROJECT_NAME)
# print(ROOT_DIR)
# ~/Desktop/TF_RL