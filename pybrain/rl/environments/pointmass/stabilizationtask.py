__author__ = 'Oswald Berthold, bertolos@informatik.hu-berlin.de'

from pybrain.rl.environments import EpisodicTask
from .pointmass import PointMassEnvironment

import numpy as np
import rospy
from std_msgs.msg import Float32MultiArray

class StabilizationTask(EpisodicTask):
    """Stabilize a 2nd order system under biased noise"""
    def __init__(self, env=None, maxsteps=1000):
        if env == None:
            env = PointMassEnvironment()
        EpisodicTask.__init__(self, env)
        self.N = maxsteps
        self.t = 0
        self.err = 0.

        # self.sensor_limits = [(-10, 10), (-100, 100), (-100, 100)]
        self.sensor_limits = [None] * self.env.outdim
        # loop over more motor channels

        # self.actor_limits = [(-1., 1.8)]
        self.action = [0. for i in range(self.env.outdim)]
        self.actor_limits = None
        
        # ROS init
        rospy.init_node("pm")
        # ROS pub/sub
        self.pub_pos    = rospy.Publisher("/robot/0/pos", Float32MultiArray)
        self.pub_tgt    = rospy.Publisher("/robot/0/target", Float32MultiArray)
        self.pub_motor  = rospy.Publisher("/robot/0/motor", Float32MultiArray)
        self.pub_reward = rospy.Publisher("/robot/0/reward", Float32MultiArray)
        self.msg_pos  = Float32MultiArray()

    def reset(self):
        """Reset task"""
        EpisodicTask.reset(self)
        self.t = 0

    def performAction(self, action):
        self.t += 1
        # print "stab action", action
        self.action = action
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
        # if self.err > 100.:
        #     print "stabilizationtask.py:isFinished, err too large", self.err
        #     return True
        if self.t >= self.N: # maximum number of steps reached
            return True
        return False
        # pass

    def getReward(self):
        target = self.env.target # -0.4
        sensors = self.env.getSensors()
        pos = self.env.getPosition()
        vel = self.env.getVelocity()
        err = target - pos

        self.err = np.abs(err)
            
        # direct continuous reward
        if err > 0:
            reward = vel
        else:
            reward = -vel

        # reward = -np.abs(err)
        # print "(pos, target) =", pos, target
        # print "stabilizationtask.py:getReward:err", err
        # reward = -np.sum(np.abs(target - pos))
        # reward = -np.sum(np.square(target - pos))
        # if np.abs(err) <= 0.1:
        #     reward = 1
        # else:
        #     reward = -1
        # print "stabilizationtask.py:getReward:reward", reward

        # publish state
        # self.msg_pos.data = [0 for i in range(6)]
        # self.msg_pos.data[0] = self.sensors[0]
        # self.msg_pos.data[3] = self.sensors[1]
        self.msg_pos.data = sensors
        self.pub_pos.publish(self.msg_pos)
        # publish action
        self.msg_pos.data = []
        self.msg_pos.data = [self.action]
        self.pub_motor.publish(self.msg_pos)
        # publish target
        self.msg_pos.data = [target]
        self.pub_tgt.publish(self.msg_pos)
        # publish reward
        self.msg_pos.data = [reward]
        self.pub_reward.publish(self.msg_pos)
        
        return np.clip(reward, -100, 100)

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
