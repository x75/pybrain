__author__ = 'Oswald Berthold, bertolos@informatik.hu-berlin.de'

from pybrain.rl.environments import EpisodicTask
from .arm import ArmEnvironment

import numpy as np
import rospy
from std_msgs.msg import Float32, Float32MultiArray

class CartesianTask(EpisodicTask):
    """Drive n-joint arm to cartesian end-effector position"""
    def __init__(self, env=None, maxsteps=1000):
        if env == None:
            env = ArmEnvironment()
        EpisodicTask.__init__(self, env)
        self.N = maxsteps
        self.t = 0
        self.err = 0.

        # self.sensor_limits = [(-10, 10), (-100, 100), (-100, 100)]
        self.sensor_limits = [None] * self.env.outdim
        # loop over more motor channels

        self.action = [0. for i in range(self.env.outdim)]
        self.actor_limits = [(-3., 3) for i in range(self.env.indim)]
        # self.actor_limits = None
        
        # ROS init
        rospy.init_node("arm")
        # ROS pub/sub
        self.pub_pos    = rospy.Publisher("/robot/0/pos", Float32MultiArray)
        self.pub_tgt    = rospy.Publisher("/robot/0/target", Float32MultiArray)
        self.pub_motor  = rospy.Publisher("/robot/0/motor", Float32MultiArray)
        self.pub_reward = rospy.Publisher("/robot/0/reward", Float32MultiArray)
        self.msg_pos  = Float32MultiArray()
        self.sub_ctrl_target = rospy.Subscriber("/robot/0/ctrl/target", Float32, self.sub_cb_ctrl)

    def sub_cb_ctrl(self, msg):
        """Set learning parameters"""
        topic = msg._connection_header["topic"].split("/")[-1]
        # print "topic", topic
        # print msg
        if topic == "target":
            self.target = msg.data
            print("target", self.target)

    def pub_all(self, sensors, action, target, reward):
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
        
    def reset(self):
        """Reset task"""
        EpisodicTask.reset(self)
        self.t = 0

    def performAction(self, action):
        self.t += 1
        print "cartesian action", action
        self.action = action
        EpisodicTask.performAction(self, action)

    def isFinished(self):
        # print self.t
        # performance is good enough
        return False

    def getReward(self):
        if self.t % 1000 == 0:
            self.env.settarget()
            print "stabilizationtask:new target", self.env.target
        # target = 0. # for e.g. velocity control
        target = self.env.target # -0.4
        sensors = self.env.getSensors()

        # pos = sensors[0] # self.env.getPosition()
        # vel = self.env.getVelocity()
        err = target - sensors
        # err = pos

        self.err = np.abs(err)
        
        reward = -np.sum(np.square(err))
        # reward = -np.sum(np.square(vel))
        
        print "(sensors, target) =", sensors, target
        # print "stabilizationtask.py:getReward:err", err
        # reward = -np.sum(np.abs(target - pos))
        # reward = -np.sum(np.square(target - pos))
        # if np.abs(err) <= 0.1:
        #     reward = 1
        # else:
        #     reward = -1
        # print "stabilizationtask.py:getReward:reward", reward

        self.pub_all(sensors, self.action, target, reward)
        
        return np.clip(reward, -100, 100)
