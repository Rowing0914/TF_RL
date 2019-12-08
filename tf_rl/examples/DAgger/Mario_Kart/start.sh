#!/bin/bash

read -p "Which controller are you using? Choose: 1(N64) or 2(PS4): " controller_type

if [ $controller_type -eq 1 ]
	then python3.6 utils.py --experiment 1 --num_episodes 2 --record_verbose 1 --controller_type N64
elif [ $controller_type -eq 2 ]
	then python3.6 utils.py --experiment 1 --num_episodes 2 --record_verbose 1 --controller_type PS4
else
	echo Please choose N64 or PS4!!
fi