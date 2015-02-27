from pybrain.rl.environments import Environment

from explauto import Environment as explEnvironment
from explauto.environment.simple_arm import SimpleArmEnvironment
from explauto.environment.simple_arm.simple_arm import joint_positions, lengths

import time
import numpy as np

class ArmEnvironment(Environment):
    indim = 3
    outdim = 4

    def __init__(self, len_episode=1000):
        self.arm = explEnvironment.from_configuration('simple_arm', 'low_dimensional')
        self.arm.noise = 0.0
        # print dir(self.arm)
        # else:
        #     self.arm.noise = 0.02
        # print dir(self.arm)
        print "arm: length ratio", self.arm.length_ratio
        print "arm: ndims", self.arm.conf.m_ndims
        print "arm: lengths", self.arm.lengths
        self.settarget()

        self.sensors = np.zeros((self.outdim,1))
        self.pos     = np.zeros((2,1))
        # self.action = np.zeros((self.indim,))

        self.reset()
        self.ti = 0
        # self.dt = 0.01
        
    def settarget(self):
        # self.target = np.random.uniform(-5, 0)
        tgt0 = np.random.uniform(np.pi/8., 3*np.pi/8.) # -0.5 - 1.0
        tgt1 = np.random.uniform(0.8, 1.0) # -0.5 - 1.0
        # tgt1 = 1.
        self.target = np.zeros((self.outdim/2, 1))
        self.target[0] = tgt1 * np.cos(tgt0)
        self.target[1] = tgt1 * np.sin(tgt0)
        # self.target = np.random.uniform(-1, 1, (self.outdim, 1))
        
    def reset(self):
        self.action = np.random.uniform(np.pi/8., 3*np.pi/8.,(self.indim,))
        self.action[1] = 0. # np.random.uniform(-np.pi, np.pi)
        self.action[2] = 0.
        # self.arm.reset()
        # pass

    def getJointPositions(self):
        (x1, x2) = joint_positions(self.action, lengths(3, 3.))
        return (x1, x2)

    def getSensors(self):
        return self.sensors.reshape((self.outdim))

    def performAction(self, action):
        # print "arm:performAction", action
        self.action += np.asarray(action)
        for i in range(self.indim):
            if self.action[i] > np.pi:
                self.action[i] -= 2*np.pi
            elif self.action[i] < -np.pi:
                self.action[i] += 2*np.pi
        # self.action[0] = np.clip(self.action[0], 0, np.pi/2)
        self.step()
    
    def step(self):
        self.pos = self.arm.compute_sensori_effect(self.action.flatten())
        # self.sensors[0,0] = self.pos[0]
        # self.sensors[1,0] = self.pos[1]
        # print __name__, self.action
        self.sensors[0,0] = self.action[0] / np.pi
        self.sensors[1,0] = self.action[1] / np.pi
        # self.sensors[self.outdim/2:] = (self.target.T - self.pos).T
        self.sensors[self.outdim/2:] = self.target
        time.sleep(0.01)
        
    def getPosition(self):
        # return self.sensors[0:2].reshape((2))
        return self.pos
