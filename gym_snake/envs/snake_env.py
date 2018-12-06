#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : gym_snake/gym_snake/envs/snake_env.py
# Author            : Denis N Dumoulin <denis.dumoulin@uclouvain.be>
# Date              : 04.12.2018
# Last Modified Date: 06.12.2018
import math
import numpy as np
import random

import gym
from gym import spaces, logger
from gym.utils import seeding
from gym.envs.classic_control import rendering

class SnakeEnv(gym.Env):
    """A snake Environment."""
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 5
    }

    def __init__(self):
        """Initialize the environment."""
        origin = np.array([1, 0])
        self.snake = Snake(origin)
        self.block_size = 20
        self.prey = np.array([])
        self.world_width = 12
        self.world_height = 12 
        self.action_space = spaces.Discrete(3)
        low = max(-self.world_width, -self.world_height)*np.ones(6)
        high = max(self.world_width, self.world_height)*np.ones(6)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def oob(self, x, y):
        """Check if '[x, y]' is out of bound."""
        return x >= .5*self.world_width \
            or y >= .5*self.world_height \
            or x < -.5*self.world_width \
            or y < -.5*self.world_height

    def get_state(self):
        """
        Obtain the state array: 
            closest wall in l-a-r directions;
            hamming distance between prey and head.
        """
        a_direction = self.snake.blocks[0] - self.snake.old_head
        l_direction = np.array(rotate(*a_direction, .5*math.pi))
        r_direction = np.array(rotate(*a_direction, -.5*math.pi))

        wall_state = np.zeros(3)
        if self.prey.tolist():
            diffe = self.prey - self.snake.blocks[0]
            l_p = diffe[np.nonzero(l_direction)] / l_direction[np.nonzero(l_direction)]
            a_p = diffe[np.nonzero(a_direction)] / a_direction[np.nonzero(a_direction)]
            r_p = diffe[np.nonzero(r_direction)] / r_direction[np.nonzero(r_direction)]
            prey_state = np.array([l_p, a_p, r_p])
        else:
            prey_state = max(self.world_width, self.world_height)*np.ones(3)

        j = 0
        for v in [l_direction, a_direction, r_direction]:
            is_empty = True
            i = 0
            while is_empty:
                i += 1
                x, y = self.snake.blocks[0] + i*v
                if ([x, y] in self.snake.blocks[3:].tolist()) or self.oob(x, y):
                    is_empty = False
            wall_state[j] = i
            j += 1

        return np.append(wall_state, prey_state)

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state

        # Create a prey with probability 1%
        if not self.prey.tolist():
            px = np.float(random.randint(-.5*self.world_width, .5*self.world_width - 1))
            py = np.float(random.randint(-.5*self.world_height, .5*self.world_height - 1))
            while [px, py] in self.snake.blocks.tolist():
                px = np.float(random.randint(-.5*self.world_width, .5*self.world_width - 1))
                py = np.float(random.randint(-.5*self.world_height, .5*self.world_height - 1))

            self.prey = np.array([px, py])
            logger.info("[INFO] -- New Prey at {}, {} ".format(px,py))
            

        # print(self.snake.blocks[0].tolist()) 
        if self.snake.blocks[0].tolist() in [self.prey.tolist()]:
            self.snake.eat_and_move(action)
            self.state = np.array([self.get_state()])
            self.prey = np.array([])
            logger.info("[INFO] -- Manger")
            reward = 500.
        else:
            self.snake.move(action)
            reward = -.5
            self.state = np.array([self.get_state()])
        
        done = self.snake.is_dead or self.oob(*self.snake.blocks[0])

        if done:
            logger.warn("DONE")
            if self.steps_beyond_done is None:
                self.steps_beyond_done = 0
                reward = -1000
            else:
                if self.steps_beyond_done == 0:
                    logger.warn("You are calling 'step()' but it's already done !")
                self.steps_beyond_done += 1
        return self.state, reward, done, {}

    def render(self, mode='human'):
        bs = self.block_size
        screen_width = self.world_width*bs + 2*bs
        screen_height = self.world_height*bs + 2*bs

        if not self.viewer:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            left_bazel = rendering.FilledPolygon([(0, 0),
                                                (0, screen_height - bs),
                                                (bs, screen_height - bs), 
                                                (bs, 0)]) 
            top_bazel = rendering.FilledPolygon([(0, screen_height - bs),
                                        (0, screen_height),
                                        (screen_width - bs, screen_height), 
                                        (screen_width - bs, screen_height - bs)]) 
            right_bazel = rendering.FilledPolygon([(screen_width - bs, 0),
                                       (screen_width - bs, screen_height),                         
                                       (screen_width, screen_height), 
                                       (screen_width, 0)])
            bottom_bazel = rendering.FilledPolygon([(bs, 0),
                                                (bs, bs),
                                                (screen_width - bs, bs), 
                                                (screen_width - bs, 0)]) 
            left_bazel.set_color(.8, .8, .8)
            top_bazel.set_color(.8, .8, .8)
            right_bazel.set_color(.8, .8, .8)
            bottom_bazel.set_color(.8, .8, .8)
            self.viewer.add_geom(left_bazel)
            self.viewer.add_geom(top_bazel)
            self.viewer.add_geom(right_bazel)
            self.viewer.add_geom(bottom_bazel)

        for b in self.snake.blocks:
            xb, yb =  b*bs + np.array([.5*screen_width,.5*screen_height])
            rb = rendering.FilledPolygon([(xb, yb), (xb, yb+bs),
                                        (xb+bs, yb+bs), (xb+bs, yb)])
            rb.set_color(0,.8, .8)
            self.viewer.add_onetime(rb)

        if self.prey.tolist():
            xp, yp =  self.prey*bs + np.array([.5*screen_width,.5*screen_height])
            rp = rendering.FilledPolygon([(xp, yp), (xp, yp+bs),
                                    (xp+bs, yp+bs), (xp+bs, yp)])
            rp.set_color(0., 0., 0.)
            self.viewer.add_onetime(rp)

        if self.state is None:
            return None

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        random.seed(seed)
        return [seed]

    def reset(self):
        self.snake = Snake(np.array([1, 0]))
        self.prey = np.array([])
        self.state = np.array([self.get_state()])
        self.steps_beyond_done = None
        return self.state 

    def close(self):
        """Close the renderer."""
        if self.viewer:
            self.viewer.close()
            self.viewer = None

class Snake():
    """A snake made of an array of blocks."""
    def __init__(self, origin):
        """Initialize a snake going to the right from 'origin'."""
        diff = np.array([1., 0.])
        self.old_head = np.array(origin - diff)
        self.blocks = np.array([origin, origin-diff, origin - 2.*diff])
        self.is_dead = False

    def eat_and_move(self, a):
         """Eat stuff under the head and move."""
         tail = self.blocks[-1]
         self.move(a)
         if not self.is_dead:
             self.blocks = np.vstack((self.blocks, tail))

    def move(self, a):
        """Move the snake one step in direction 'a'."""
        assert a in [0, 1, 2]

        if not self.is_dead:
            diff = np.array(self.blocks[0] - self.old_head)

            if a == 0:
                dxdy = diff
            elif a == 1:
                dxdy = np.array(rotate(*diff, .5*math.pi))
            else:
                dxdy = np.array(rotate(*diff, -.5*math.pi))

            self.blocks = np.roll(self.blocks, 2)
            self.blocks[0] = self.blocks[1] + dxdy
            self.old_head = self.blocks[1]
        else:
            return

        if self.blocks[0].tolist() in self.blocks[1:].tolist():
            self.is_dead = True
        logger.info("[INFO] -- Head moved to {}".format(self.blocks[0]))

    def __repr__(self):
        return str(self.blocks)

def rotate(X, Y, theta):
    x = np.round(X*math.cos(theta) - Y*math.sin(theta))
    y = np.round(X*math.sin(theta) + Y*math.cos(theta))
    return x, y
