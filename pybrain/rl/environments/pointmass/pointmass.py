from pybrain.rl.environments.environment import Environment
from ode_inert_system import InertParticle2D

import rospy
import numpy as np
from std_msgs.msg import Float32MultiArray
import time

class PointMassEnvironment(Environment):
    indim = 1
    outdim = 3

    # continuous domain

    # some parameters
    dt = 1e-1
    alag = 1
    pm_mass = 1.
    forcefield = False

    def __init__(self, len_episode=1000):
        # ROS init
        rospy.init_node("pm")
        self.ip2d = InertParticle2D(a0 = np.zeros((1, self.indim)),
                                    v0 = np.zeros((1, self.indim)),
                                    x0 = np.zeros((1, self.indim)),
                                    numsteps = len_episode+1, dt = self.dt,
                                    dim = self.indim,
                                    alag = self.alag-1,
                                    mass = self.pm_mass,
                                    forcefield = self.forcefield)
        self.ip2d.anoise_mean = -0.05
        self.target = np.random.uniform(-1, 1)
        # ros pub/sub
        self.pub_pos    = rospy.Publisher("/robot/0/pos", Float32MultiArray)
        self.pub_tgt    = rospy.Publisher("/robot/0/target", Float32MultiArray)
        self.pub_motor  = rospy.Publisher("/robot/0/motor", Float32MultiArray)
        self.msg_pos  = Float32MultiArray()

        self.reset()
        self.u = np.zeros((self.indim, 1))
        self.sensors = np.zeros((self.outdim, 1))
        self.ti = 0

    def reset(self):
        a0 = np.zeros((1, self.indim))
        # v0 = np.zeros((1, self.indim))
        v0 = np.random.uniform(-0.1, 0.1, (1, self.indim))
        x0 = np.zeros((1, self.indim))
        self.ti = 0
        self.ip2d.reset(x0, v0, a0)

    def getSensors(self):
        self.sensors = np.vstack((self.ip2d.x[self.ti,:], self.ip2d.v[self.ti,:], self.ip2d.a[self.ti,:] * 0.))
        # return np.asarray(self.sensors)
        self.msg_pos.data = [0 for i in range(6)]
        self.msg_pos.data[0] = self.sensors[0]
        self.msg_pos.data[3] = self.sensors[1]
        self.pub_pos.publish(self.msg_pos)
        self.msg_pos.data = [self.target, 0., 0.]
        self.pub_tgt.publish(self.msg_pos)
        # time.sleep(0.001)
        return self.sensors.reshape((self.outdim))

    def performAction(self, action):
        # print "pm action pre", action
        # action = (action / 21.) * 2
        # action = (action / 7.) * 2
        # print "pm action post", action
        # self.u = action
        self.ip2d.u[self.ti] = action
        self.msg_pos.data = [action]
        self.pub_motor.publish(self.msg_pos)
        # print "self.u", self.u
        self.step()

    def step(self):
        self.ip2d.step(self.ti, self.dt)
        self.ti += 1

    def getPosition(self):
        return self.ip2d.x[self.ti]
