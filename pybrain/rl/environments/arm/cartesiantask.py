__author__ = 'Oswald Berthold, bertolos@informatik.hu-berlin.de'

from pybrain.rl.environments import EpisodicTask
from .arm import ArmEnvironment

import numpy as np
import rospy
from std_msgs.msg import Float32, Float32MultiArray
from smp_msgs.srv import SingleValFloat

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
        # ROS pub/sub/srv
        self.pub_pos    = rospy.Publisher("/robot/0/pos", Float32MultiArray)
        self.pub_tgt    = rospy.Publisher("/robot/0/target", Float32MultiArray)
        self.pub_pos1   = rospy.Publisher("/robot/1/pos", Float32MultiArray)
        self.pub_tgt1   = rospy.Publisher("/robot/1/target", Float32MultiArray)
        self.pub_motor  = rospy.Publisher("/robot/0/motor", Float32MultiArray)
        self.pub_reward = rospy.Publisher("/robot/0/reward", Float32MultiArray)
        self.msg_pos    = Float32MultiArray()
        self.sub_ctrl_target = rospy.Subscriber("/robot/0/ctrl/target", Float32, self.sub_cb_ctrl)
        self.sub_ctrl_theta  = rospy.Subscriber("/robot/0/ctrl/theta", Float32, self.sub_cb_ctrl)
        self.srv_numbodies   = rospy.Service("%s/numbodies" % ("Experiment"), SingleValFloat, self.srv_cb_numbodies)

    def srv_cb_numbodies(self, req):
        return 2
    
    def sub_cb_ctrl(self, msg):
        """Set learning parameters"""
        topic = msg._connection_header["topic"].split("/")[-1]
        print "topic", topic
        # print msg
        if topic == "target":
            self.target = msg.data
            print("target", self.target)
        elif topic == "theta":
            print "cartesian:", dir(self.env)
            # self.target = msg.data
            # print("target", self.target)

    def pub_all(self, sensors, action, target, reward):
        # publish end-effector pos / state
        self.msg_pos.data = [0 for i in range(3)]
        self.msg_pos.data[0] = sensors[0]
        self.msg_pos.data[1] = sensors[1]
        # self.msg_pos.data = sensors
        self.pub_pos.publish(self.msg_pos)
        # publish action
        # self.msg_pos.data = []
        self.msg_pos.data = self.action.tolist()
        self.pub_motor.publish(self.msg_pos)
        # publish target
        # print type(target), target.shape
        self.msg_pos.data = target.flatten().tolist()
        self.msg_pos.data.append(0.)
        self.pub_tgt.publish(self.msg_pos)
        # publish reward
        # print type(reward)
        self.msg_pos.data = [reward]
        self.pub_reward.publish(self.msg_pos)
        # publish first joint position
        self.msg_pos.data = [0 for i in range(3)]
        (x1, x2) = self.env.getJointPositions()
        self.msg_pos.data[0] = x1[0]
        self.msg_pos.data[1] = x2[0]
        self.pub_pos1.publish(self.msg_pos)
        # publish dummy target
        self.msg_pos.data = [0, 0, 0]
        self.pub_tgt1.publish(self.msg_pos)
        
    def reset(self):
        """Reset task"""
        EpisodicTask.reset(self)
        self.t = 0

    def performAction(self, action):
        self.t += 1
        self.action = action
        # self.action[-2] = 0.
        self.action[-1] = 0.
        # print "cartesian action", self.action
        EpisodicTask.performAction(self, action)

    def isFinished(self):
        # print self.t
        # performance is good enough
        return False

    def getReward(self):
        # set new target
        if self.t % 1000 == 0:
            self.env.settarget()
            print "cartesiantask:new target", self.env.target

        reward = 0.
        
        # target = 0. # for e.g. velocity control
        target = self.env.target # -0.4
        sensors = self.env.getSensors()
        # print "cartesiantask.py:sensor", sensors
        # print "cartesiantask.py:target", target
        pos = self.env.getPosition()
        # vel = self.env.getVelocity()
        # err = target.T - sensors[0:2]
        err = target.T - pos
        # err = pos
        # print "cartesiantask.py:error", err

        self.err = np.sum(np.abs(err))
        
        # reward = -np.sum(np.square(err))
        # reward = -np.sum(np.square(vel))
        # reward = -self.err
        if self.err < 0.5:
            reward = 1.
        else:
            reward = -1.
        
        # print "cartesiantask:(sensors, target) =", sensors, target
        # print "stabilizationtask.py:getReward:err", err
        # reward = -np.sum(np.abs(target - pos))
        # reward = -np.sum(np.square(target - pos))
        # if np.abs(err) <= 0.1:
        #     reward = 1
        # else:
        #     reward = -1
        # print "stabilizationtask.py:getReward:reward", reward

        # self.pub_all(sensors, self.action, target, reward)
        self.pub_all(pos, self.action, target, reward)
        
        return np.clip(reward, -100, 100)
