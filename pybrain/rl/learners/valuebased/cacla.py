__author__ = "Oswald Berthold"

import numpy as np
from pybrain.rl.learners.valuebased.valuebased import ValueBasedLearner

import rospy
from std_msgs.msg import Float32MultiArray

class CACLA(ValueBasedLearner):
    """Continuous Actor-Critic Learning Automaton (CACLA) algorithm

    From Hasselt09"""

    def __init__(self, module, task, alpha=1e-2, beta=1e-2, gamma=1e-1, explorer=None):
        ValueBasedLearner.__init__(self)

        self.module = module
        self.task   = task
        self.explorer = explorer

        self.annealT = 20000.

        # learning rates for V and A
        self.alpha = alpha # action lr
        self.beta = beta  # value lr
        self.gamma = gamma  # horizon
                
        # what's called laststate here is lastobs in agent parlance
        self.laststate = None
        self.lastaction = None

        # print "self.module", self.module
        # rospy.init_node("pm1")
        # self.pub_Astate = rospy.Publisher("/robot/0/Astate", Float32MultiArray)
        # self.msg_Astate = Float32MultiArray()
        self.pub_V = rospy.Publisher("/robot/0/V", Float32MultiArray)
        self.msg_V = Float32MultiArray()
        self.msg_V.data = [0 for i in range(3)]
        self.cnt = 0
        # print dir(self.explorer)
        self.sigma_ = self.explorer.sigma
        print "cacla:sigma", self.explorer.sigma
        self.alpha_ = alpha
        self.beta_  = beta

    def anneal(self):
        # self.explorer._setSigma(self.explorer.sigma * 0.99999)
        # print self.cnt
        if self.cnt > 20000:
            self.alpha = 0.
            self.beta = 0.
            self.explorer._setSigma(np.ones_like(self.explorer.sigma) * 0.001)
        else:
            self.alpha = self.alpha_ / (1 + (self.cnt / self.annealT))
            self.beta = self.beta_ / (1 + (self.cnt / self.annealT))
            self.explorer._setSigma(self.explorer.sigma * 0.99999) # _ / (1 + (self.cnt / self.annealT)))
        # print __name__, self.explorer.sigma, self.alpha, self.beta
        # pass
    
    def learn(self):
        # print self.dataset
        samples = [[self.dataset.getSample()]]
        # print "calca.py:learn:t", self.task.t
        # print dir(self.explorer)
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
                Vtstp1 = self.module.getValue(state)
                Vtst   = self.module.getValue(self.laststate)
                if np.isnan(Vtst) or np.isnan(Vtstp1):
                    print "NAN"
                    sys.exit()
                
                Vtarget = reward + self.gamma * Vtstp1
                # Vtarget = self.lastreward + self.gamma * Vtstp1
                # Vtarget = (1-self.gamma) * reward + self.gamma * Vtstp1
                
                self.module.trainV(self.beta, Vtarget, Vtstp1, self.laststate)
                # print __name__, "r", reward, "Vt-Vt_", Vtarget - Vtstp1
                
                # self.module.Vw = np.clip(self.module.Vw, -10., 10.)
                Vtp1st = self.module.getValue(self.laststate)

                # if Vtp1st > Vtst:
                # if reward > self.lastreward:
                if Vtarget > Vtst:
                # if Vtstp1 > Vtst:
                    # print "learning " * 10
                    # action module learning
                    # target = self.lastaction
                    
                    # Atarget = (self.module.Astate * (self.lastaction - action)).reshape((self.module.outdim, self.module.hdim))
                    # # print "Atarget.shape", Atarget.shape, self.module.Aw.shape
                    # # Atst = np.dot(self.module.Aw, self.laststate)
                    # # delta = target - Atst
                    # # print "target", target
                    # self.module.Aw += self.alpha * Atarget # - action)

                    # lastaction = self.module.getAction(self.laststate)
                    # print action, lastaction
                    Atst   = self.module.getAction(self.laststate)
                    # print "atst", Atst, action, self.module.Astate
                    d1 = (action - Atst)
                    # print "d1", d1.shape, (d1 * self.module.Astate).shape
                    # self.module.trainA(self.alpha, action, self.lastaction, state)
                    self.module.trainA(self.alpha, action, Atst, state)
                    
                    # print "cacla |Aw|", self.cnt, np.linalg.norm(self.module.Aw)
                # else:
                #     self.module.Aw -= self.alpha * Atarget # - action)
                # self.module.Aw = np.clip(self.module.Aw, -10., 10.)

                # print "norms", np.linalg.norm(self.module.Aw), np.linalg.norm(self.module.Vw)
                self.msg_V.data[0] = Vtst
                self.msg_V.data[1] = Vtp1st # Vtstp1
                self.msg_V.data[2] = Vtarget
                self.pub_V.publish(self.msg_V)
                # self.msg_Astate.data = self.module.Astate.flatten().tolist()
                # # print self.msg_Astate.data
                # self.pub_Astate.publish(self.msg_Astate)
                    
                # move state to oldstate
                self.laststate = state
                self.lastaction = action
                self.lastreward = reward
                # self.Vlast = self.Vcurr
                # self.Alast = self.Acurr
            self.cnt += 1
            self.anneal()


class CACLA2(ValueBasedLearner):
    """Continuous Actor-Critic Learning Automaton (CACLA) algorithm

    From Hasselt09"""

    def __init__(self, module, task, alpha=1e-2, beta=1e-2, gamma=1e-1, explorer=None):
        ValueBasedLearner.__init__(self)

        self.module = module
        self.task   = task
        self.explorer = explorer

        # what's called laststate here is lastobs in agent parlance
        self.laststate = None
        self.lastaction = None

        # rospy.init_node("pm1")
        self.pub_V = rospy.Publisher("/robot/0/V", Float32MultiArray)
        self.msg_V = Float32MultiArray()
        self.msg_V.data = [0 for i in range(3)]
        self.cnt = 0
    
    def learn(self):
        # print __name__, type(self.dataset)
        # print __name__, dir(self.dataset)
        # print __name__, self.dataset
        (state, action, reward) = self.dataset.getSample()
        seq = self.dataset.getSequence(0)
        # print __name__, seq[0].shape
        # print __name__, "state", seq[0]
        # print __name__, "action", seq[1]
        # print __name__, "reward", seq[2]
        
        if self.laststate == None:
            self.lastaction = action
            self.laststate = state
            self.lastreward = reward
            return

        if self.cnt < 2:
            self.cnt += 1
            return
        elif self.cnt > 1001:
            sl = slice(self.cnt-1000, self.cnt)
        else:
            sl = slice(None, None)
        print __name__, "state, value", seq[0].shape, seq[2].shape
        print __name__, "state, value sl", seq[0][sl].shape, seq[2][sl].shape
        self.module.trainV(0, seq[2][sl], None, seq[0][sl])
        Vtst = self.module.getValue(state)
        

        if reward > self.lastreward:
            # print "bumm"
            self.module.trainA(0, seq[1][sl], None, seq[0][sl])
                

        # print "norms", np.linalg.norm(self.module.Aw), np.linalg.norm(self.module.Vw)
        self.msg_V.data[0] = Vtst
        self.msg_V.data[1] = 0. # Vtp1st # Vtstp1
        self.msg_V.data[2] = 0. # Vtarget
        self.pub_V.publish(self.msg_V)
        
        # move state to oldstate
        self.laststate = state
        self.lastaction = action
        self.lastreward = reward
        # self.Vlast = self.Vcurr
        # self.Alast = self.Acurr
        self.cnt += 1
