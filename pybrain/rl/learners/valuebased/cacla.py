__author__ = "Oswald Berthold"

import numpy as np
from pybrain.rl.learners.valuebased.valuebased import ValueBasedLearner

import rospy
from std_msgs.msg import Float32MultiArray

class CACLA(ValueBasedLearner):
    """Continuous Actor-Critic Learning Automaton (CACLA) algorithm

    From Hasselt09"""

    def __init__(self, module, task, alpha=1e-2, beta=1e-2, gamma=1e-1):
        ValueBasedLearner.__init__(self)

        self.module = module
        self.task   = task

        # learning rates for V and A
        self.alpha = alpha # action lr
        self.beta = beta  # value lr
        self.gamma = gamma  # horizon
                
        # what's called laststate here is lastobs in agent parlance
        self.laststate = None
        self.lastaction = None

        # print "self.module", self.module
        # rospy.init_node("pm1")
        self.pub_V = rospy.Publisher("/robot/0/V", Float32MultiArray)
        self.msg_V = Float32MultiArray()
        self.msg_V.data = [0 for i in range(3)]
        
    def learn(self):
        samples = [[self.dataset.getSample()]]

        for seq in samples:
            # print "seq", seq
            # get state, performed action, got reward
            for state, action, reward in seq:
                # print "cacla.py:learn:state", state
                # print "cacla.py:learn:action", action
                # print "cacla.py:learn:reward", reward
                # first learning call has no last state: skip
                if self.laststate == None:
                    self.lastaction = action
                    self.laststate = state
                    self.lastreward = reward
                    continue

                
                # # do it
                # # value module learning
                # # print "cacla.py:learn:reward", reward
                # print "self.module", self.module
                Vtst   = self.module.getValue(self.laststate)
                Vtstp1 = self.module.getValue(state)
                if np.isnan(Vtst) or np.isnan(Vtstp1):
                    print "NAN"
                    sys.exit()
                # print "Vtst, Vtstp1", Vtst, Vtstp1
                # target = self.lastreward + self.gamma * Vtstp1
                # if self.task.isFinished():
                #     target = reward
                # else:
                Vtarget = reward + self.gamma * Vtstp1
                # print "self.lastreward", self.lastreward
                # print "gamma * V_t(s_t+1)", self.gamma * Vtstp1
                # print "target V", target
                # delta = target - Vtst
                # print "self.module.Vstate.shape", self.module.Vstate.shape
                delta = (Vtarget  - Vtstp1) * self.module.Vstate
                # print "|delta|", np.linalg.norm(delta)
                # print "delta Vw", delta * self.laststate
                # print "self.module.Vw", self.module.Vw
                # # if v_t+1(s_t) > V_t(s_t) then
                
                # y = np.tanh(np.dot(self.module.Vwin, self.laststate))
                # print "y.shape", y.shape
                # Vtp1st = np.dot(self.module.Vw, y)
                
                # Vtp1st = np.dot(self.module.Vw, self.laststate)
                # value learn
                self.module.Vw += self.beta * delta # * self.laststate
                # self.module.Vw = np.clip(self.module.Vw, -10., 10.)
                # if Vtp1st > Vtst:
                # if reward > self.lastreward:
                if Vtarget > Vtst:
                    # if Vtstp1 > Vtst:
                    # print "learning " * 10
                    # action module learning
                    # target = self.lastaction
                    Atarget = (self.module.Astate * (self.lastaction - action)).reshape((self.module.hdim,))
                    # print "Atarget.shape", Atarget.shape, self.module.Aw.shape
                    # Atst = np.dot(self.module.Aw, self.laststate)
                    # delta = target - Atst
                    # print "target", target
                    self.module.Aw += self.alpha * Atarget # - action)
                    # self.module.Aw = np.clip(self.module.Aw, -10., 10.)

                # print "norms", np.linalg.norm(self.module.Aw), np.linalg.norm(self.module.Vw)
                self.msg_V.data[0] = Vtst
                self.msg_V.data[1] = Vtstp1
                self.msg_V.data[2] = Vtarget
                self.pub_V.publish(self.msg_V)
                    
                # move state to oldstate
                self.laststate = state
                self.lastaction = action
                self.lastreward = reward
                # self.Vlast = self.Vcurr
                # self.Alast = self.Acurr
