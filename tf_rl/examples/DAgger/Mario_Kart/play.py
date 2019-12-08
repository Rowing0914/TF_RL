#!/usr/bin/env python

# to remove unnecessary path from python path
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')


from utils import Data_bot
from termcolor import cprint
from skimage.transform import resize
# don't delete gym_mupen64plus!!
import gym, gym_mupen64plus
from train import create_model
import numpy as np

# image sizes of input X
IMG_W = 200
IMG_H = 66
IMG_D = 3


# Play
class Actor(object):

    def __init__(self):
        # Load in model from train.py and load in the trained weights
        self.model = create_model(keep_prob=1) # no dropout
        self.model.load_weights('model_weights.h5')

        # Init contoller for manual override
        self.real_controller = Data_bot()

    def get_action(self, obs):

        ## Look
        im = resize(obs, (IMG_H, IMG_W, IMG_D))
        vec = im.reshape((IMG_H, IMG_W, IMG_D))
        vec = np.expand_dims(vec, axis=0) # expand dimensions for predict, it wants (1,66,200,3) not (66, 200, 3)
        ## Think
        joystick = self.model.predict(vec, batch_size=1)[0]


        ## Act

        ### calibration
        output = [
            int(joystick[0] * 80),
            int(joystick[1] * 80),
            int(round(joystick[2])),
            int(round(joystick[3])),
            int(round(joystick[4])),
        ]

        cprint("Manual: " + str(output), 'yellow')

        return output


    def get_action_expert_advice(self, obs):

        ### determine manual override
        manual_override = self.real_controller.LeftBumper == 1

        if not manual_override:
            ## Look
            im = resize(obs, (IMG_H, IMG_W, IMG_D))
            vec = im.reshape((IMG_H, IMG_W, IMG_D))
            vec = np.expand_dims(vec, axis=0) # expand dimensions for predict, it wants (1,66,200,3) not (66, 200, 3)
            ## Think
            joystick = self.model.predict(vec, batch_size=1)[0]

        else:
            joystick = self.real_controller.read()
            joystick[1] *= -1 # flip y (this is in the config when it runs normally)


        ## Act

        ### calibration
        output = [
            int(joystick[0] * 80),
            int(joystick[1] * 80),
            int(round(joystick[2])),
            int(round(joystick[3])),
            int(round(joystick[4])),
        ]

        ### print to console
        if manual_override:
            cprint("Manual: " + str(output), 'yellow')
        else:
            cprint("AI: " + str(output), 'green')

        return output


if __name__ == '__main__':
    env = gym.make('Mario-Kart-Royal-Raceway-v0')
    # env = gym.make('Mario-Kart-Luigi-Raceway-v0')

    obs = env.reset()
    env.render()
    print('env ready!')

    actor = Actor()
    print('actor ready!')

    print('beginning episode loop')
    total_reward = 0
    end_episode = False
    while not end_episode:
        action = actor.get_action(obs)
        obs, reward, end_episode, info = env.step(action)
        env.render()
        total_reward += reward

    print('end episode... total reward: ' + str(total_reward))

    obs = env.reset()
    print('env ready!')

    input('press <ENTER> to quit')

    env.close()
