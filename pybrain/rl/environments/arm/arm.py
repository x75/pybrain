from pybrain.rl.environments import Environment

from explauto import Environment as explEnvironment
from explauto.environment.simple_arm import SimpleArmEnvironment
from explauto.environment.simple_arm.simple_arm import joint_positions, lengths

import numpy as np

class ArmEnvironment(Environment):
    indim = 3
    outdim = 2

    def __init__(self, len_episode=1000):
        self.arm = explEnvironment.from_configuration('simple_arm', 'low_dimensional')
        self.arm.noise = 0.0
        # else:
        #     self.arm.noise = 0.02
        # print dir(self.arm)
        print "arm: length ratio", self.arm.length_ratio
        print "arm: ndims", self.arm.conf.m_ndims
        self.settarget()

        self.sensors = np.zeros((self.outdim,1))
        self.action = np.zeros((self.indim,1))

        self.reset()
        self.ti = 0
        
    def settarget(self):
        # self.target = np.random.uniform(-5, 0)
        self.target = np.random.uniform(-1, 1, (self.outdim, 1))
        
    def reset(self):
        # self.arm.reset()
        pass

    def getSensors(self):
        return self.sensors.reshape((self.outdim))

    def performAction(self, action):
        print "arm:performAction", action
        self.action = np.asarray(action)
        self.step()
    
    def step(self):
        self.sensors = self.arm.compute_sensori_effect(self.action.flatten())
