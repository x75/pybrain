__author__ = 'Oswald Berthold, bertolos@informatik.hu-berlin.de'

from pybrain.rl.environments import EpisodicTask
from .pointmass import PointMassEnvironment

import numpy as np

class StabilizationTask(EpisodicTask):
    """Stabilize a 2nd order system under biased noise"""
    def __init__(self, env=None, maxsteps=1000):
        if env == None:
            env = PointMassEnvironment()
        EpisodicTask.__init__(self, env)
        self.N = maxsteps
        self.t = 0

        # self.sensor_limits = [(-10, 10), (-100, 100), (-100, 100)]
        self.sensor_limits = [None] * self.env.outdim
        # loop over more motor channels

        # self.actor_limits = [(-1., 1.8)]
        self.actor_limits = None

    def reset(self):
        """Reset task"""
        EpisodicTask.reset(self)
        self.t = 0

    def performAction(self, action):
        self.t += 1
        # action = action - (self.numActions-1)//2.
        # action = action - (self.numActions-1)//2.
        # print "stab action", action
        EpisodicTask.performAction(self, action)

    def isFinished(self):
        # print self.t
        # performance is good enough
        if self.t >= 100:
            accerr = np.sum(np.abs(self.env.ip2d.x[self.t-100:self.t]))
            # print "accerr", accerr
            if accerr < 1.:
                return True
        # point mass is too far away
        if self.t >= self.N: # maximum number of steps reached
            return True
        return False
        # pass

    def getReward(self):
        target = self.env.target # -0.4
        pos = self.env.getPosition()
        vel = self.env.getVelocity()
        err = target - pos
        
        # direct continuous reward
        if err > 0:
            reward = -vel
        else:
            reward = vel

        # reward = np.abs(err)
        # print "(pos, target) =", pos, target
        print "stabilizationtask.py:getReward:err", err
        # reward = -np.sum(np.abs(target - pos))
        # reward = -np.sum(np.square(target - pos))
        # if np.abs(err) <= 0.1:
        #     reward = 1
        # else:
        #     reward = -1
        # print "stabilizationtask.py:getReward:reward", reward
        return np.clip(reward, -10, 10)

class DiscreteStabilizationTask(StabilizationTask):
    numActions = 3
    # numActions = 21
    # numActions = 7
    def __init__(self, env=None, maxsteps=1000):
        if env == None:
            env = PointMassEnvironment()
        EpisodicTask.__init__(self, env)
        self.N = maxsteps
        self.t = 0
        self.sensor_limits = [None] * self.env.outdim
        # loop over more motor channels

        # self.actor_limits = [(-1., 1.8)]
        self.actor_limits = None

    def reset(self):
        """Reset task"""
        EpisodicTask.reset(self)
        self.t = 0
        
    def getObservation(self):
        """get measurements"""
        sensors = self.env.getSensors()
        if self.sensor_limits:
            sensors = self.normalize(sensors)
        return sensors
    
    def performAction(self, action):
        # self.t += 1
        action = action - (self.numActions-1)//2.
        StabilizationTask.performAction(self, action)

    def getReward(self):
        target = self.env.target # -0.4
        pos = self.env.getPosition()
        # print "(pos, target) =", pos, target
        err = target - pos
        # reward = -np.sum(np.abs(target - pos))
        reward = -np.sum(np.square(target - pos))
        # if np.abs(err) <= 0.1:
        #     reward = 1
        # else:
        #     reward = -1
        print "stabilizationtask.py:getReward:reward", reward
        return reward

    def isFinished(self):
        # print self.t
        # performance is good enough
        if self.t >= 100:
            accerr = np.sum(np.abs(self.env.ip2d.x[self.t-100:self.t]))
            print "accerr", accerr
            if accerr < 0.1:
                return True
        # point mass is too far away
        if self.t >= self.N: # maximum number of steps reached
            return True
        return False
        # pass
