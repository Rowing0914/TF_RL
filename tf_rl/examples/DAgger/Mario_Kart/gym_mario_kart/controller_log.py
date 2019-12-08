from pynput.keyboard import Key, Listener

temp = list()

ACTION = {
	Key.up: [0,0,1,0,0],
    Key.left: [-10,0,1,0,0],
    Key.right: [10,0,1,0,0]
    }

def on_press(key):
	if key in [Key.up, Key.left, Key.right]:
		temp.append(ACTION[key])
		print(ACTION[key])

def on_release(key):
	if key == Key.esc:
		# Stop listener
		return False

# Collect events until released
with Listener(on_press=on_press, on_release=on_release) as listener:
	listener.join()

# save the log
import numpy as np
print(np.array(temp).shape)
np.savetxt("play_log.csv", np.array(temp), delimiter=",")