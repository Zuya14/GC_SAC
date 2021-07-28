# import sys

import gym
# import numpy as np
# import gym.spaces
from gym.utils import seeding
# import torch

import pybullet as p
import cv2
import math

import random
# import copy

import bullet_lidar
# import sim  

from sim_abs import sim_abs
import numpy as np

class sim_square3(sim_abs):

    def calcInitPos(self, initPos=None):
        if initPos is None:
            initPos = np.random.rand(2) * 8.5
            initPos = initPos + 0.25 

            return np.concatenate([initPos, [0.0]])
        else:
            return np.array(initPos)

    def calcTgtPos(self, tgtPos):
        if tgtPos is None:
            tgtPos = np.random.rand(2) * 8.5
            tgtPos = tgtPos + 0.25 

            return np.concatenate([tgtPos, [0.0]])
        else:
            return np.array(tgtPos)

    def loadObstacle(self):

        for i in range(10):
            self.bodyUniqueIds += [self.phisicsClient.loadURDF("urdf/wall.urdf", basePosition=(i-1,  9, 0))]
            self.bodyUniqueIds += [self.phisicsClient.loadURDF("urdf/wall.urdf", basePosition=(  i, -1, 0))]
            self.bodyUniqueIds += [self.phisicsClient.loadURDF("urdf/wall.urdf", basePosition=( -1,i-1, 0))]
            self.bodyUniqueIds += [self.phisicsClient.loadURDF("urdf/wall.urdf", basePosition=(  9,  i, 0))]

    ''' action = [v, cos, sin, theta] '''
    def calcAction(self, action):
        theta = np.arctan2(action[2], action[1])

        _, ori = self.getRobotPosInfo()
        yaw = p.getEulerFromQuaternion(ori)[2]
        dtheta = yaw + theta
        
        self.vx = action[0] * np.sin(dtheta)
        self.vy = action[0] * np.cos(dtheta)
        # self.w = action[3]
        self.w = 0.0

    def isArrive(self):
        return  np.linalg.norm(self.tgt_pos - self.getState(), ord=2) < 0.1
        
    def render(self):
        w = 800
        h = 800
        img = np.zeros((w, h, 3), np.uint8)
        center = (w//2, h//2)

        points = [
            [0.0, 0.0],
            [0.0, 9.0],
            [9.0, 0.0],
            [9.0, 9.0],
            [3.0, 0.0],
            [3.0, 6.0],
            [6.0, 6.0],
            [6.0, 0.0],
        ]

        points = np.array(points)
        points += np.array([-4.5, -4.5])

        pts = np.zeros((points.shape[0], 2))

        k = center[0] if w<h else center[1]
        # k = 10
        maxLen = 5

        for a, b in zip(pts, points):
            a[0] =  b[0] * (k/maxLen)
            a[1] = -b[1] * (k/maxLen)

        pts += center
        pts = pts.astype(np.int)

        cv2.line(img, tuple(pts[0]), tuple(pts[1]), color=(255,255,255), thickness=1, lineType=cv2.LINE_8, shift=0)
        cv2.line(img, tuple(pts[0]), tuple(pts[2]), color=(255,255,255), thickness=1, lineType=cv2.LINE_8, shift=0)
        cv2.line(img, tuple(pts[1]), tuple(pts[3]), color=(255,255,255), thickness=1, lineType=cv2.LINE_8, shift=0)
        cv2.line(img, tuple(pts[2]), tuple(pts[3]), color=(255,255,255), thickness=1, lineType=cv2.LINE_8, shift=0)

        # print([self.observe(), tgtpos])
        points2 = np.array([self.getState()[:2], self.tgt_pos[:2]])
        points2 += np.array([-4.5, -4.5])
        pts2 = np.zeros((2, 2))

        for a, b in zip(pts2, points2):
            a[0] =  b[0] * (k/maxLen)
            a[1] = -b[1] * (k/maxLen)

        pts2 += center
        pts2 = pts2.astype(np.int)

        cv2.circle(img, tuple(pts2[0]), radius=20, color=(255,0,0), thickness=-1, lineType=cv2.LINE_8, shift=0)
        cv2.circle(img, tuple(pts2[1]), radius=20, color=(0,0,255), thickness=2, lineType=cv2.LINE_8, shift=0)

        return img


class square3Env(gym.Env):
    global_id = 0

    def __init__(self):
        super().__init__()
        self.seed(seed=random.randrange(10000))
        self.sim = None

        self.name = "square3Env"

    def setting(self, _id=-1, mode=p.DIRECT, sec=0.1):
        if _id == -1:
            self.sim = sim_square3(square3Env.global_id, mode, sec)
            square3Env.global_id += 1
        else:
            self.sim = sim_square3(_id, mode, sec)

        # self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        self.lidar = self.createLidar()

        self.state_space = gym.spaces.Box(low=0.0, high=9.0, shape=(3,))
        self.velocity_space = gym.spaces.Box(low=0.0, high=9.0, shape=(3,))
        self.observation_space = gym.spaces.Box(low=0.0, high=9.0, shape=(self.lidar.shape[0],))

        self.sec = sec

        self._max_episode_steps = 500
        # self._max_episode_steps = 100
        
        self.setParams()

        self.reset()

    def copy(self, _id=-1):
        new_env = square3Env()
        
        if _id == -1:
            new_env.sim = self.sim.copy(square3Env.global_id)
            square3Env.global_id += 1
        else:
            new_env.sim = self.sim.copy(_id)

        new_env.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        new_env.lidar = new_env.createLidar()

        new_env.state_space = gym.spaces.Box(low=0.0, high=9.0, shape=(3,))
        new_env.velocity_space = gym.spaces.Box(low=0.0, high=9.0, shape=(3,))
        new_env.observation_space = gym.spaces.Box(low=0.0, high=9.0, shape=(self.lidar[0],))

        new_env.sec = self.sec

        return new_env

    def reset(self, initpos=None, tgtpos=None):
        assert self.sim is not None, print("call setting!!") 
        self.sim.reset(sec=self.sec, initPos=initpos, tgtPos=tgtpos)
        return self.observe_all()

    def test_reset(self):
        return self.reset(initpos=[1.5, 1.5, 0.0], tgtpos=[7.0, 1.5, 0.0])

    def createLidar(self):
        resolusion = 90
        # resolusion = 12
        # resolusion = 36
        # resolusion = 1
        deg_offset = 90.
        rad_offset = deg_offset*(math.pi/180.0)
        startDeg = -180. + deg_offset
        endDeg = 180. + deg_offset

        # maxLen = 20.
        # maxLen = 10.
        maxLen = 5.
        minLen = 0.
        return bullet_lidar.bullet_lidar(startDeg, endDeg, resolusion, maxLen, minLen)

    def step(self, action):

        done = self.sim.step(action)

        state, velocity, observation = self.observe_all()

        reward = self.get_reward()

        done = done or (self.sim.steps == self._max_episode_steps)

        return [state, velocity, observation], reward, done, {}

    def get_left_steps(self):
        return self._max_episode_steps - self.sim.steps

    def observe_all(self):
        return self.sim.getState(), self.sim.getVelocity(), self.sim.getObserve(self.lidar)

    def get_reward(self):
        return self.calc_reward(self.sim.isContacts(), self.sim.getState(), self.sim.tgt_pos, self.sim.getOldState())

    def calc_reward(self, contact, pos, tgt_pos, old_pos):
        rewardContact = -self.params['contact'] if contact else 0.0

        d1 = np.linalg.norm(tgt_pos - old_pos, ord=2)
        d2 = np.linalg.norm(tgt_pos - pos, ord=2)

        # rewardDistance = - 0.1 * np.linalg.norm(tgt_pos - pos, ord=2)
        rewardDistance = (not contact) * self.params['distance'] * (d1 - d2) 
        
        rewardDistance += - self.params['log_distance'] * np.log(1.0 + d2)

        reward = rewardContact + rewardDistance

        return reward
    def setParams(self):
        self.params = {
            'contact': 10.0,
            'distance': 1.0/self.sec,
            'log_distance': 0.1,
            }

    def getParams(self):
        return self.params

    def getEnvParams(self):
        return {
            'lidar_points': self.lidar._shape[0],
            'lidar_max': self.lidar.maxLen,
            'lidar_min': self.lidar.minLen,
            }

    def render(self, mode='human', close=False):
        # return self.sim.render(self.lidar)
        return self.sim.render()

    def close(self):
        self.sim.close()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def sample_random_action(self):
        return self.action_space.sample()

    def getState(self):
        return self.sim.getState()


if __name__ == '__main__':
    
    env = square3Env()
    env.setting()

    i = 0

    while True:
        i += 1
        
        action = np.array([1.0, 1.0, 0.0, 0.0])

        state, _, done, _ = env.step(action)

        # print(state)

        cv2.imshow("env", env.render())
        if done or cv2.waitKey(100) >= 0:
            print(i)
            break