# from __future__ import division, print_function
#
# import sys
# import numpy
# import gym
# import time
# from optparse import OptionParser
#
# import gym_minigrid
#
# def main():
#     parser = OptionParser()
#     parser.add_option(
#         "-e",
#         "--env-name",
#         dest="env_name",
#         help="gym environment to load",
#         default='MiniGrid-MultiRoom-N6-v0'
#     )
#     (options, args) = parser.parse_args()
#
#     # Load the gym environment
#     env = gym.make(options.env_name)
#
#     def resetEnv():
#         env.reset()
#         if hasattr(env, 'mission'):
#             print('Mission: %s' % env.mission)
#
#     resetEnv()
#
#     # Create a window to render into
#     renderer = env.render('human')
#
#     def keyDownCb(keyName):
#         if keyName == 'BACKSPACE':
#             resetEnv()
#             return
#
#         if keyName == 'ESCAPE':
#             sys.exit(0)
#
#         action = 0
#
#         if keyName == 'LEFT':
#             action = env.actions.left
#         elif keyName == 'RIGHT':
#             action = env.actions.right
#         elif keyName == 'UP':
#             action = env.actions.forward
#
#         elif keyName == 'SPACE':
#             action = env.actions.toggle
#         elif keyName == 'PAGE_UP':
#             action = env.actions.pickup
#         elif keyName == 'PAGE_DOWN':
#             action = env.actions.drop
#
#         elif keyName == 'RETURN':
#             action = env.actions.done
#
#         else:
#             print("unknown key %s" % keyName)
#             return
#
#         obs, reward, done, info = env.step(action)
#
#         print('step=%s, reward=%.2f' % (env.step_count, reward))
#
#         if done:
#             print('done!')
#             resetEnv()
#
#     renderer.window.setKeyDownCb(keyDownCb)
#
#     while True:
#         env.render('human')
#         time.sleep(0.01)
#
#         # If the window was closed
#         if renderer.window == None:
#             break
#
# if __name__ == "__main__":
#     main()

import gym
import gym_sokoban
import time
from PIL import Image
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description='Run environment with random selected actions.')
parser.add_argument('--rounds', '-r', metavar='rounds', type=int,
                    help='number of rounds to play (default: 1)', default=1)
parser.add_argument('--steps', '-s', metavar='steps', type=int,
                    help='maximum number of steps to be played each round (default: 300)', default=300)
parser.add_argument('--env', '-e', metavar='env',
                    help='Environment to load (default: Sokoban-v0)', default='Sokoban-v0')
parser.add_argument('--save', action='store_true',
                    help='Save images of single steps')
parser.add_argument('--gifs', action='store_true',
                    help='Generate Gif files from images')
parser.add_argument('--render_mode', '-m', metavar='render_mode',
                    help='Render Mode (default: human)', default='human')

args = parser.parse_args()
env_name = args.env
n_rounds = args.rounds
n_steps = args.steps
save_images = args.save or args.gifs
generate_gifs = args.gifs
render_mode = args.render_mode

# Creating target directory if images are to be stored
if save_images and not os.path.exists('images'):
    try:
        os.makedirs('images')
    except OSError:
        print('Error: Creating images target directory. ')

ts = time.time()
env = gym.make(env_name)
ACTION_LOOKUP = env.unwrapped.get_action_lookup()
print("Created environment: {}".format(env_name))


def print_available_actions():
    """
    Prints all available actions nicely formatted..
    :return:
    """
    available_actions_list = []
    for i in range(len(ACTION_LOOKUP)):
        available_actions_list.append(
            'Key: {} - Action: {}'.format(i, ACTION_LOOKUP[i])
        )
    display_actions = '\n'.join(available_actions_list)
    print()
    print('Action out of Range!')
    print('Available Actions:\n{}'.format(display_actions))
    print()


for i_episode in range(n_rounds):
    print('Starting new game!')
    observation = env.reset()

    for t in range(n_steps):
        env.render(render_mode)

        action = input('Select action: ')
        try:
            action = int(action)

            if not action in range(len(ACTION_LOOKUP)):
                raise ValueError

        except ValueError:
            print_available_actions()
            continue

        observation, reward, done, info = env.step(action)
        print(ACTION_LOOKUP[action], reward, done, info)

        if save_images:
            img = Image.fromarray(np.array(observation), 'RGB')
            img.save(os.path.join('images', 'observation_{}_{}.png'.format(i_episode, t)))

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            env.render(render_mode)
            break

    if generate_gifs:
        print('')
        import imageio

        with imageio.get_writer(os.path.join('images', 'round_{}.gif'.format(i_episode)), mode='I', fps=1) as writer:

                for t in range(n_steps):
                    try:

                        filename = os.path.join('images', 'observation_{}_{}.png'.format(i_episode, t))
                        image = imageio.imread(filename)
                        writer.append_data(image)

                    except:
                        pass

env.close()
time.sleep(10)