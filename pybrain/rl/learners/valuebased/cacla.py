__author__ = "Oswald Berthold"

import numpy as np
from pybrain.rl.learners.valuebased.valuebased import ValueBasedLearner

class CACLA(ValueBasedLearner):
    """Continuous Actor-Critic Learning Automaton (CACLA) algorithm

    From Hasselt09"""

    def __init__(self, module, alpha=1e-2, beta=1e-2, gamma=1e-1):
        ValueBasedLearner.__init__(self)

        self.module = module

        # learning rates for V and A
        self.alpha = alpha # action lr
        self.beta = beta  # value lr
        self.gamma = gamma  # horizon
                
        # what's called laststate here is lastobs in agent parlance
        self.laststate = None
        self.lastaction = None

        # print "self.module", self.module
        
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
                target = self.lastreward + self.gamma * Vtstp1
                # print "self.lastreward", self.lastreward
                # print "gamma * V_t(s_t+1)", self.gamma * Vtstp1
                # print "target V", target
                delta = target - Vtst
                # print "|delta|", np.linalg.norm(delta)
                # print "delta Vw", delta * self.laststate
                self.module.Vw += self.beta * delta # * self.laststate
                self.module.Vw = np.clip(self.module.Vw, -100., 100.)
                # print "self.module.Vw", self.module.Vw
                # # if v_t+1(s_t) > V_t(s_t) then
                Vtp1st = np.dot(self.module.Vw, self.laststate)
                # if Vtp1st > Vtst:
                if reward > self.lastreward:
                    # if Vtstp1 > Vtst:
                    # print "learning" * 10
                    # action module learning
                    target = self.lastaction
                    # Atst = np.dot(self.module.Aw, self.laststate)
                    # delta = target - Atst
                    # print "target", target
                    self.module.Aw += self.alpha * target
                    self.module.Aw = np.clip(self.module.Aw, -100., 100.)

                # print "norms", np.linalg.norm(self.module.Aw), np.linalg.norm(self.module.Vw)
                    
                # move state to oldstate
                self.laststate = state
                self.lastaction = action
                self.lastreward = reward
                # self.Vlast = self.Vcurr
                # self.Alast = self.Acurr
