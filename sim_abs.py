from abc import ABC, abstractmethod
import cv2
import numpy as np
import pybullet as p
import pybullet_utils.bullet_client as bc
import pybullet_data

import math

import lidar_util

robot_name = "urdf/pla-robot.urdf"

class sim_abs(ABC):

    def __init__(self, _id=0, mode=p.DIRECT, sec=0.01):
        self._id = _id
        self.mode = mode
        self.phisicsClient = bc.BulletClient(connection_mode=mode)
        self.reset(sec=sec)

    @abstractmethod
    def calcInitPos(self, initPos=None):
        if initPos is None:
            return np.zeros(3)
        else:
            return np.array(initPos)

    @abstractmethod
    def calcTgtPos(self, tgtPos):
        if tgtPos is None:
            return np.zeros(3)
        else:
            return np.array(tgtPos)

    def onRect(self, pos, rec_s, rec_e):
        return pos[0] >= rec_s[0] and pos[0] <= rec_e[0] and pos[1] >= rec_s[1] and pos[1] <= rec_e[1]


    ''' initPos = [x, y, theta], tgtPos = [x, y, theta] '''
    def reset(self, initPos=None, tgtPos=None, vx=0.0, vy=0.0, w=0.0, sec=0.01, action=None, clientReset=False):
        if clientReset:
            self.phisicsClient = bc.BulletClient(connection_mode=self.mode)

        self.steps = 0
        self.sec = sec

        self.vx = vx
        self.vy = vy
        self.w = w

        self.phisicsClient.resetSimulation()
        self.robotUniqueId = 0 
        self.bodyUniqueIds = []
        self.phisicsClient.setTimeStep(sec)

        self.action = action if action is not None else [0.0, 0.0, 0.0]
        
        self.done = False

        self.tgt_pos = self.calcTgtPos(tgtPos)

        x, y, theta = self.calcInitPos(initPos)
        self.loadBodys(x, y, theta)

        self.old_state = self.getState()

        # x, y = self.getState()[:2]
        # self.distance = math.sqrt((x - 10.0)**2 + (y - 10.0)**2)
        # self.old_distance = self.distance

    def getId(self):
        return self._id

    def loadBodys(self, x, y, theta):
        self.robotPos = (x,y,0)
        self.robotOri = p.getQuaternionFromEuler([0, 0, theta])

        self.robotUniqueId = self.phisicsClient.loadURDF(
            robot_name,
            basePosition=self.robotPos,
            baseOrientation = self.robotOri
            )

        self.loadObstacle()
    
    @abstractmethod
    def loadObstacle(self):
        pass

    @abstractmethod
    def calcAction(self, action):
        self.vx = 0
        self.vy = 0
        self.w = 0

    def step(self, action):

        self.old_state = self.getState()

        if not self.done:

            self.calcAction(action)

            self.updateRobotInfo()

            self.phisicsClient.stepSimulation()

            if self.isContacts():
                self.done = True

            if self.isArrive():
                self.done = True

        else:
            self.vx = 0
            self.vy = 0
            self.w = 0

        self.action = np.array([self.vx, self.vy, self.w])

        self.steps += 1

        return self.done

    def getObserve(self, bullet_lidar):
        pos, ori = self.getRobotPosInfo()
        yaw = p.getEulerFromQuaternion(ori)[2]
        # scanDist = bullet_lidar.scanDistance(self.phisicsClient, pos, yaw, height=0.1)
        scanDist = bullet_lidar.scanDistance(self.phisicsClient, pos, yaw, height=0.55)
        # self.scanDist = scanDist / bullet_lidar.maxLen
        self.scanDist = scanDist
        self.scanDist = self.scanDist.astype(np.float32)

        return self.scanDist

    def getState(self):
        pos, ori = self.getRobotPosInfo()
        return np.array([pos[0], pos[1], p.getEulerFromQuaternion(ori)[2]])

    def getOldState(self):
        return self.old_state

    def getVelocity(self):
        return np.array([self.vx, self.vy, self.w ])

    def render(self, bullet_lidar):
        pos, ori = self.getRobotPosInfo()
        yaw = p.getEulerFromQuaternion(ori)[2]
        scanDist = bullet_lidar.scanDistance(self.phisicsClient, pos, yaw, height=0.1)
        self.scanDist = scanDist / bullet_lidar.maxLen
        self.scanDist = self.scanDist.astype(np.float32)

        img = lidar_util.imshowLocalDistance("render"+str(self.phisicsClient), 800, 800, bullet_lidar, self.scanDist, maxLen=1.0, show=False, line=True)
        # print(self.scanDist.shape)

        return img

    def updateRobotInfo(self):

        self.phisicsClient.resetBaseVelocity(
            self.robotUniqueId,
            linearVelocity=[self.vx, self.vy, 0],
            angularVelocity=[0, 0, self.w]
            )

        self.robotPos, self.robotOri = self.phisicsClient.getBasePositionAndOrientation(self.robotUniqueId)

    def getRobotPosInfo(self):
        return self.robotPos, self.robotOri

    def close(self):
        self.phisicsClient.disconnect()

    def contacts(self):
        contactList = []
        for i in self.bodyUniqueIds[0:]: # 接触判定
            contactList += self.phisicsClient.getContactPoints(self.robotUniqueId, i)
        return contactList 

    def isContacts(self):
        return len(self.contacts()) > 0

    @abstractmethod
    def isArrive(self):
        pass
    
    def isDone(self):
        return self.done

    @abstractmethod
    def render(self):
        pass