"""
state_reshape = (1, 4)
loss_fn = "huber_loss"
grad_clip_flg = "None"
sync_freq = 1000
train_interval = 1
gamma = 0.99
update_hard_or_soft = "hard"
soft_update_tau = 1e-2
decay_type = "linear"
decay_steps = 3_000
reward_buffer_ep = 5
epsilon_start = 1.0
epsilon_end = 0.02
batch_size = 32
test_episodes = 10
"""

import os

params = [
	"--loss_fn=huber_loss --log_dir=../logs/logs/DQN_huber",
	"--loss_fn=MSE   --log_dir=../logs/logs/DQN_MSE",
	"--loss_fn=huber_loss --log_dir=../logs/logs/DQN2_huber",
	"--loss_fn=MSE   --log_dir=../logs/logs/DQN2_MSE",
	"--loss_fn=huber_loss --log_dir=../logs/logs/DQN3_huber",
	"--loss_fn=MSE   --log_dir=../logs/logs/DQN3_MSE",
	# "--loss_fn=huber --log_dir=huber",
	# "--loss_fn=huber --log_dir=huber",
	# "--loss_fn=huber --log_dir=huber",
	# "--loss_fn=huber --log_dir=huber",
]

for param in params:
	os.system("python3.6 DQN/DQN_eager.py --mode=CartPole {}".format(param))