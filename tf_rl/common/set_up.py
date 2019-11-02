import datetime
from tf_rl.common.abs_path import ROOT_DIR

def set_up_for_training(log_dir_name, env_name, seed):
    exp_date = datetime.datetime.now().strftime("%Y%m%d%H%M")

    logdir = {
        "summary_path": ROOT_DIR + "/logs/log/{}/{}-{}-seed{}".format(log_dir_name, env_name, exp_date, seed),
        "video_path": ROOT_DIR + "/logs/video/{}/{}-{}-seed{}".format(log_dir_name, env_name, exp_date, seed),
        "model_path": ROOT_DIR + "/logs/model/{}/{}-{}-seed{}".format(log_dir_name, env_name, exp_date, seed),
        "traj_path": ROOT_DIR + "/logs/traj/{}/{}-{}-seed{}".format(log_dir_name, env_name, exp_date, seed),
    }
    return logdir