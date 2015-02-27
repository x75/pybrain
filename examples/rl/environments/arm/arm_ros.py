#!/usr/bin/env python

import rospy
from std_msgs.msg import Float64MultiArray, Float32MultiArray

from explauto import Environment as explEnvironment
from explauto.environment.simple_arm import SimpleArmEnvironment
from explauto.environment.simple_arm.simple_arm import joint_positions, lengths

from pybrain.rl.environments.arm import ArmEnvironment

import time
import numpy as np


class ArmRos(object):
    actionDimension = 3
    stateDimension = 4
    gotmotors = True
    motors = np.zeros((actionDimension,))
    sensors = np.zeros((stateDimension,))

    def __init__(self, len_episode):
        self.len_episode=len_episode
        # self.arm = ArmEnvironment(len_episode=self.len_episode)
        self.arm = explEnvironment.from_configuration('simple_arm', 'low_dimensional')
        self.arm.noise = 0.0
        
        print "arm: length ratio", self.arm.length_ratio
        print "arm: ndims", self.arm.conf.m_ndims
        print "arm: lengths", self.arm.lengths
        
        self.sub_motors = rospy.Subscriber("/motors", Float64MultiArray, self.cb_motors)
        self.pub_sensors = rospy.Publisher("/sensors", Float64MultiArray, queue_size=1)
        self.sensor_msg = Float64MultiArray()
        self.pub_pos1 = rospy.Publisher("/robot/0/pos", Float32MultiArray, queue_size=1)
        self.pub_pos2 = rospy.Publisher("/robot/1/pos", Float32MultiArray, queue_size=1)
        self.pos1_msg = Float32MultiArray()
        self.rate = rospy.Rate(1000) # 10hz

        # let ros come up
        time.sleep(2)
        
    def cb_motors(self, msg):
        # print msg
        self.motors += np.asarray(msg.data) * 0.1
        self.motors[-1] = 0
        # print "motors", motors
        # clip motors
        if self.motors[0] > np.pi:
            self.motors[0] -= 2*np.pi
        if self.motors[0] < -np.pi:
            self.motors[0] += 2*np.pi
        if self.motors[1] > np.pi:
            self.motors[1] -= 2*np.pi
        if self.motors[1] < -np.pi:
            self.motors[1] += 2*np.pi
        self.gotmotors = True
        
    def run(self):
        waitcnt = 0
        while not rospy.is_shutdown():
            if self.gotmotors or waitcnt > 100:
                print "motors", self.motors
                # self.arm.performAction(self.motors)
                self.sensors[0:2] = self.arm.compute_sensori_effect(self.motors.flatten())
                self.sensors[2] = self.motors[0]
                self.sensors[3] = self.motors[1]
                (x1, x2) = joint_positions(self.motors.flatten(), lengths(3, 3.))
                # self.sensors = self.arm.getSensors()
                # sensor_msg.data = np.random.uniform(0, 1, (4,1)).flatten().tolist()
                self.sensor_msg.data = self.sensors
                # print sensor_msg.data
                self.pub_sensors.publish(self.sensor_msg)

                # visualization
                self.pos1_msg.data = [0 for i in range(3)]
                self.pos1_msg.data[0] = self.sensors[0]
                self.pos1_msg.data[1] = self.sensors[1]
                self.pub_pos1.publish(self.pos1_msg)
                # first joint
                # (x1, x2) = self.arm.getJointPositions()
                # print x1, x2
                # x1 = 1
                # x2 = 2
                self.pos1_msg.data[0] = x1[0]
                self.pos1_msg.data[1] = x2[0]
                self.pub_pos2.publish(self.pos1_msg)
                self.gotmotors = False
                waitcnt = 0
            else:
                waitcnt += 1

            self.rate.sleep()
        
        # rospy.spin()

        
def main():
    rospy.init_node("armros")
    armros = ArmRos(100000)
    armros.run()
    
if __name__ == "__main__":
    main()
