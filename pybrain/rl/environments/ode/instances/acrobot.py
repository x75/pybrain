__author__ = 'Frank Sehnke, sehnke@in.tum.de'

from pybrain.rl.environments.ode import ODEEnvironment, sensors, actuators
import imp
from scipy import array

class AcrobotEnvironment(ODEEnvironment):
    def __init__(self, renderer=True, realtime=True, ip="127.0.0.1", port="21590", buf='16384'):
        ODEEnvironment.__init__(self, renderer, realtime, ip, port, buf)
        # load model file
        modelfile = imp.find_module('pybrain')[1] + "/rl/environments/ode/models/acrobot.xode"
        print("loading acrobot form xode: %s" % modelfile)
        try:
            # self.loadXODE(imp.find_module('pybrain')[1] + "/rl/environments/ode/models/acrobot.xode")
            self.loadXODE(modelfile)
        except Exception, e:
            print("exception + %s" % e)
        print("done loading acrobot form xode")

        # standard sensors and actuators
        self.addSensor(sensors.JointSensor())
        self.addSensor(sensors.JointVelocitySensor())
        self.addActuator(actuators.JointActuator())

        #set act- and obsLength, the min/max angles and the relative max touques of the joints
        self.actLen = self.indim
        self.obsLen = len(self.getSensors())

        self.stepsPerAction = 1

if __name__ == '__main__' :
    w = AcrobotEnvironment()
    while True:
        w.step()
        if w.stepCounter == 1000: w.reset()

