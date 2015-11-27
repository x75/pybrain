from pybrain.rl.environments.environment import Environment
from smplib.ode_inert_system import InertParticle2D

import numpy as np
import time

class PointMassEnvironment(Environment):
    indim = 1
    outdim = 4 #3

    # continuous domain

    # some parameters
    dt = 5e-2
    alag = 1
    pm_mass = 1.
    forcefield = False

    def __init__(self, len_episode=1000):
        self.ip2d = InertParticle2D(a0 = np.zeros((1, self.indim)),
                                    v0 = np.zeros((1, self.indim)),
                                    x0 = np.zeros((1, self.indim)),
                                    numsteps = len_episode+1, dt = self.dt,
                                    dim = self.indim,
                                    alag = self.alag-1,
                                    mass = self.pm_mass,
                                    forcefield = self.forcefield)
        # self.ip2d.anoise_mean = -0.05
        self.ip2d.anoise_mean = -0.5
        self.settarget()

        self.reset()
        self.u = np.zeros((self.indim, 1))
        self.sensors = np.zeros((self.outdim, 1))
        self.ti = 0

    def settarget(self):
        # self.target = np.random.uniform(-5, 0)
        self.target = np.random.uniform(-12, 12)

    def reset(self):
        a0 = np.zeros((1, self.indim))
        # v0 = np.zeros((1, self.indim))
        v0 = np.random.uniform(-0.1, 0.1, (1, self.indim))
        x0 = np.zeros((1, self.indim))
        self.ti = 0
        self.ip2d.reset(x0, v0, a0)

    def getSensors(self):
        # print "pointmass:getSensors"
        # self.sensors = np.vstack((self.ip2d.x[self.ti,:], self.ip2d.v[self.ti,:], self.ip2d.a[self.ti,:] * 0.))
        self.sensors = np.vstack((self.ip2d.x[self.ti,:], self.ip2d.v[self.ti,:], self.ip2d.a[self.ti,:] * 0., self.target))
        # self.sensors = np.vstack((self.target - self.ip2d.x[self.ti,:], self.ip2d.v[self.ti,:], self.ip2d.a[self.ti,:] * 0.))
        return self.sensors.reshape((self.outdim))

    def performAction(self, action):
        # print "pm action pre", action
        # action = (action / 21.) * 2
        # action = (action / 7.) * 2
        # print "pm action post", action
        # self.u = action
        self.ip2d.u[self.ti] = action
        # print "self.u", self.u
        self.step()

    def step(self):
        if self.ti == 0:
            # print "ti", self.ti
            self.settarget()
        self.ip2d.step(self.ti, self.dt)
        time.sleep(0.01)
        self.ti += 1

    def getPosition(self):
        return self.ip2d.x[self.ti]
    
    def getVelocity(self):
        return self.ip2d.v[self.ti]
