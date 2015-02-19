__author__ = "Oswald Berthold"

import numpy as np
from pybrain.rl.learners.valuebased.valuebased import ValueBasedLearner

class CACLA(ValueBasedLearner):
    """Continuous Actor-Critic Learning Automaton (CACLA) algorithm

    From Hasselt09"""

    def __init__(self, actionmodule):
        ValueBasedLearner.__init__(self)

        self.actionmodule = actionmodule

        # learning rates for V and A
        self.alpha = 1e-3 # action lr
        self.beta = 1e-3  # value lr

        # V approximation
        self.V = np.zeros((3, 1))
        # A approximation
        self.A = np.zeros((3, 1))
                
        # what's called laststate here is lastobs in agent parlance
        self.laststate = None
        self.lastaction = None
        
    def learn(self):
        samples = [[self.dataset.getSample()]]

        for seq in samples:
            # print "seq", seq
            # get state, performed action, got reward
            for state, action, reward in seq:
                print "cacla.py:learn:state", state
                print "cacla.py:learn:action", action
                print "cacla.py:learn:reward", reward
                # first learning call has no last state: skip
                if self.laststate == None:
                    self.lastaction = action
                    self.laststate = state
                    self.lastreward = reward
                    continue

                # do it
                # value module learning
                # self.module.updateValue()
                # if v_t+1(s_t) > V_t(s_t) then
                # action module learning
                
                # move state to oldstate
                self.laststate = state
                self.lastaction = action
                self.lastreward = reward
