import os
import datetime
from tf_rl.common.abs_path import ROOT_DIR as ROOT


def set_up_for_training(env_name, seed, gpu_id, log_dir="Test", prev_log="", google_colab=False):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if google_colab:
        # mount your drive on google colab
        from google.colab import drive
        drive.mount("/content/gdrive")

    if prev_log == "":
        exp_date = datetime.datetime.now().strftime("%Y%m%d%H%M%S")  # dir_name format

        logdir = {
            "summary_path": ROOT + "/logs/logs/{}/{}-{}-seed{}".format(log_dir, env_name, exp_date, seed),
            "video_path": ROOT + "/logs/videos/{}/{}-{}-seed{}".format(log_dir, env_name, exp_date, seed),
            "model_path": ROOT + "/logs/models/{}/{}-{}-seed{}".format(log_dir, env_name, exp_date, seed),
            "traj_path": ROOT + "/logs/trajs/{}/{}-{}-seed{}".format(log_dir, env_name, exp_date, seed),
            "controller_path": ROOT + "/logs/controllers/{}/{}-{}-seed{}".format(log_dir, env_name, exp_date, seed),
        }
    else:
        # check if the specified previous training dir exists
        assert os.path.isdir(ROOT + "/logs/logs/{}".format(prev_log)), "Previous logs not found!!"
        assert os.path.isdir(ROOT + "/logs/videos/{}".format(prev_log)), "Previous video not found!!"
        assert os.path.isdir(ROOT + "/logs/models/{}".format(prev_log)), "Previous model not found!!"
        assert os.path.isdir(ROOT + "/logs/trajs/{}".format(prev_log)), "Previous trajs not found!!"
        assert os.path.isdir(ROOT + "/logs/controllers/{}".format(prev_log)), "Previous controller not found!!"

        logdir = {
            "summary_path": ROOT + "/logs/logs/{}".format(prev_log),
            "video_path": ROOT + "/logs/videos/{}".format(prev_log),
            "model_path": ROOT + "/logs/models/{}".format(prev_log),
            "traj_path": ROOT + "/logs/trajs/{}".format(prev_log),
            "controller_path": ROOT + "/logs/controllers/{}".format(prev_log)
        }
    return logdir
