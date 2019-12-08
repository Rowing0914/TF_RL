"""
Author: Norio Kosaka

==== joystick input ====
x        = self.JoystickX
y        = self.JoystickY
forward  = self.A
jump     = self.RightBumper
use_item = self.C_left

==== Joystick output vector ====
[x, y, forward, jump, use_item]

"""

from PIL import ImageTk, Image
from inputs import get_gamepad
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import mss, time, math, threading, sys, argparse, os, collections

import warnings
warnings.filterwarnings("ignore")

# ====== Global variables definitions ======
# window position
SRC_W = 640
SRC_H = 480
SRC_D = 3

# joystick related variables
OFFSET_X = 0
OFFSET_Y = 0
MAX_JOY_VAL_N64 = math.pow(2,8)
MAX_JOY_VAL_PS4 = math.pow(2,15)

# image sizes of input X
IMG_W = 200
IMG_H = 66
IMG_D = 3

# file location
OUTPUTDIR = "./temp/"
INPUT_DATA_CSV = "./temp/data.csv"

# skip frame rate
SKIP_FRAME = 2

# ====== Class Definition ======
class Data_bot():
	"""
	This module has the funtionality of taking screenshot and recording the inputs from the controller
	"""
	def __init__(self, record_verbose=1, controller_type="PS4"):
		self.record_verbose = record_verbose
		self.controller_type = controller_type
		self._t = 0
		self.num_episode = 0
		self.screen_capture = mss.mss()
		self.outfile = open(INPUT_DATA_CSV, 'a')
		self.JoystickX = 0
		self.JoystickY = 0
		self.A = 0
		self.RightBumper = 0
		self.C_left = 0
		self._monitor_thread = threading.Thread(target=self._monitor_controller, args=())
		self._monitor_thread.daemon = True
		self._monitor_thread.start()
		self._img = collections.deque(maxlen=2)
		self.X = list()
		self.y = list()

	def _monitor_controller(self):
		"""
		Using threading, a spawned process keeps maintaining the state of this class
		Then, whenever read_data is called, we can get the current information(capture and inputs from the controller)
		"""
		if self.controller_type == "N64":
			while True:
				events = get_gamepad()
				for event in events:
					# remove noise
					if ((event.code == 'ABS_Z') and (event.state in [130,131,132,133,134,135,136,137,138,139])) or (event.code == 'SYN_REPORT'):
						pass
					else:
						# joystick: right(237) or left(30)
						if event.code == 'ABS_Z':
							self.JoystickX = event.state / MAX_JOY_VAL_N64 # normalise data between 0 and 1
						elif event.code == 'ABS_Y':
							self.JoystickY = event.state / MAX_JOY_VAL_N64 # normalise data between 0 and 1
						elif (event.code == 'MSC_SCAN') and (event.state == 589831):
							self.A = 1
						elif (event.code == 'MSC_SCAN') and (event.state == 589830):
							self.RightBumper = 1
						elif (event.code == 'MSC_SCAN') and (event.state == 589826):
							self.C_left = 1
		elif self.controller_type == "PS4":
			while True:
				events = get_gamepad()
				for event in events:
					if event.code == 'ABS_X':
						self.JoystickX = event.state / MAX_JOY_VAL_PS4 # normalise data between 0 and 1
					elif event.code == 'ABS_Y':
						self.JoystickY = event.state / MAX_JOY_VAL_PS4 # normalise data between 0 and 1
					elif event.code == 'BTN_SOUTH':
						self.A = 1
					elif event.code == 'BTN_TR':
						self.RightBumper = 1
					elif event.code == 'BTN_EAST':
						self.C_left = 1

	def _buffer_reset(self):
		"""
		After reading the data, we refresh the buffer
		"""
		self.JoystickX = 0
		self.JoystickY = 0
		# self.A = 0
		self.RightBumper = 0
		self.C_left = 0

	def dev_screen_shot(self):
		"""
		take a screenshot of the defined region on the screen
		"""
		sct_img = self.screen_capture.grab({
			"top"   : OFFSET_Y,
			"left"  : OFFSET_X,
			"width" : SRC_W,
			"height": SRC_H
			})

		# Create the Image
		return Image.frombytes('RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX')

	def screen_shot(self):
		"""
		take a screenshot of the defined region on the screen
		"""
		sct_img = self.screen_capture.grab({
			"top"   : OFFSET_Y,
			"left"  : OFFSET_X,
			"width" : SRC_W,
			"height": SRC_H
			})

		# return the resized numpy array represents the image
		im = np.array(Image.frombytes('RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX').getdata())
		return resize(im, (IMG_H, IMG_W, IMG_D))

	def dev_save_data(self):
		"""
		Save data into outputDir
		"""
		image_file = OUTPUTDIR + 'img_' + str(self.t) + '.png'
		self.img.save(image_file)

		# write csv line
		self.outfile.write( image_file + ',' + ','.join(map(str, self.controller_data)) + '\n' )

	def save_data(self):
		"""
		Save data into outputDir
		"""
		print("\n==== SAVING TO FILE... ====")
		self.X = np.asarray(self.X)
		self.y = np.asarray(self.y)

		# directly save the data in numpy array format
		np.save("data/X_ep_{0}".format(self.num_episode), self.X)
		np.save("data/y_ep_{0}".format(self.num_episode), self.y)
		
		# keep the track of the number of episode
		self.num_episode += 1

		# refresh the memory
		self.X = list()
		self.y = list()
		print("==== DONE ====")

	def read_data(self):
		"""
		Read data
		"""
		x        = self.JoystickX
		y        = self.JoystickY
		forward  = self.A
		jump     = self.RightBumper
		use_item = self.C_left

		if self.record_verbose:
			print(x, y, forward, jump, use_item)
		
		self._buffer_reset()
		return [x, y, forward, jump, use_item]

	def dev_main(self):
		"""
		Main loop
		"""
		self.img = self.dev_screen_shot()
		self.controller_data = self.read_data()
		self.dev_save_data()

	def main(self):
		"""
		Main loop with skiping frame
		"""
		# we store the data once in N(SKIP_FRAME) steps
		if self._t == SKIP_FRAME:
			self._image = np.max(np.stack(self._img), axis=0)
			self.controller_data = self.read_data()

			# store a numpy array of an image and y
			self.X.append(self._image)
			self.y.append(self.controller_data)
			self._t = 0

		# put the image to the queue which has the length of 2
		self._img.append(self.screen_shot())
		self._t += 1


# ====== Utility functions ======
def record(bot):
	try:
		while True:
			bot.main()
	except KeyboardInterrupt:
		bot.save_data()

def check_raw_input():
	"""
	investigate the raw_input from the controller
	"""
	while True:
		events = get_gamepad()
		for event in events:
			print(event.code, event.state)


# ====== Dev purpose functions ======
def dev_record(record_verbose):
	bot = Data_bot(record_verbose)
	while True:
		bot.dev_main()

def dev_prepare():
	X, y = list(), list()

	# load sample
	image_files = np.loadtxt(INPUT_DATA_CSV, delimiter=',', dtype=str, usecols=(0,))
	joystick_values = np.loadtxt(INPUT_DATA_CSV, delimiter=',', usecols=(1,2,3,4,5))

	# add joystick values to y
	y.append(joystick_values)

	# load, prepare and add images to X
	_t = 0
	image = collections.deque(maxlen=2)
	for image_file in image_files:
		image = imread(image_file)
		_t += 1
		if _t == SKIP_FRAME:
			# take maximum value of two successive frames
			image = np.max(np.stack(image), axis=0)
			im = resize(image, (IMG_H, IMG_W, IMG_D))
			vec = im.reshape((IMG_H, IMG_W, IMG_D))
			X.append(vec)
			_t = 0

	print("==== SAVING TO FILE... ====")
	X = np.asarray(X)
	y = np.concatenate(y)

	# directly save the data in numpy array format
	np.save("data/X", X)
	np.save("data/y", y)

	print("==== DONE ====")

def dev_check_data():
	"""
	check the basic stats of the recorded data
	"""
	import pandas as pd

	df = pd.read_csv(INPUT_DATA_CSV)
	df.columns = ['path','x', 'y', 'f', 'j', 'u']
	print(df.describe())

def dev_clean_up(num_episode):
	"""
	garbage collection function.
	Before moving on to the another episode, we remove the all files in `temp`
	"""
	# os.system("rm -rf temp")
	# os.system("mkdir temp")
	os.system("mv ./data/X.npy ./data/X_ep_{0}.npy".format(num_episode))
	os.system("mv ./data/y.npy ./data/y_ep_{0}.npy".format(num_episode))
	print("HOLD ON A SECOND")
	time.sleep(1)

def aggregate(num_episodes):
	print("==== AGGREGATION: START ====")
	for i in range(num_episodes):
		if i == 0:
			X = np.load("data/X_ep_{0}.npy".format(i))
			y = np.load("data/y_ep_{0}.npy".format(i))
		else:
			temp_X = np.load("data/X_ep_{0}.npy".format(i))
			temp_y = np.load("data/y_ep_{0}.npy".format(i))
			np.concatenate((X, temp_X), axis=0)
			np.concatenate((y, temp_y), axis=0)
	os.system("rm ./data/X_*")
	os.system("rm ./data/y_*")
	np.save("data/X", X)
	np.save("data/y", y)

	print("==== AGGREGATION: DONE ====")


if __name__ == '__main__':
	# refresh the directory
	os.system("rm ./data/*")

	# define args for this program
	parser = argparse.ArgumentParser()
	parser.add_argument("--experiment", help="Record the demo and store the outcome to `temp` directory")
	parser.add_argument("--controller_type", help="What kind of controller are you using? Choose : N64 or PS4")
	parser.add_argument("--num_episodes", help="how many times you would like to record?")
	parser.add_argument("--record_verbose", help="flag to specify if you would like to see the verbose of inputs")
	parser.add_argument("--check_raw_input", help="check the raw input from the current gamepad")
	args = parser.parse_args()

	if args.experiment == "0":
		for i in range(int(args.num_episodes)):
			try:
				print("""==== Episode: {0}/{1} ====\nTo start recording: PRESS Enter\nTo stop recording : PRESS Ctrl + c""".format(i+1, int(args.num_episodes)))
				_in = input()
				print("==== START RECORDING ====")
				dev_record(args.record_verbose)
			except KeyboardInterrupt:
				print("==== FINISH RECORDING ====")
				print("==== START PREPARING DATA ====")
				dev_prepare()
				dev_clean_up(i)
		aggregate(int(args.num_episodes))
	elif args.experiment == "1":
		bot = Data_bot(record_verbose=args.record_verbose, controller_type=args.controller_type)
		for i in range(int(args.num_episodes)):
			try:
				print("""==== Episode: {0}/{1} ====\nTo start recording: PRESS Enter\nTo stop recording : PRESS Ctrl + c""".format(i+1, int(args.num_episodes)))
				_in = input()
				print("==== START RECORDING ====")
				record(bot)
			except KeyboardInterrupt:
				print("==== FINISH RECORDING ====")
				pass
		aggregate(int(args.num_episodes))
	elif args.check_raw_input == "1":
		check_raw_input()
	else:
		print('hello')