#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : gym_snake/gym_snake/__init__.py
# Author            : Denis N Dumoulin <denis.dumoulin@uclouvain.be>
# Date              : 04.12.2018
# Last Modified Date: 04.12.2018
from gym.envs.registration import register

register(
        id='Snake-v0',
        entry_point='gym_snake.envs:SnakeEnv'
        )
