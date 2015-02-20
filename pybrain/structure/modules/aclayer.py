__author__ = 'Oswald Berthold'

from pybrain.structure.modules.linearlayer import LinearLayer
from pybrain.structure.modules.module import Module

import numpy as np


class ACLayer(Module):
    """Module combining value (critic) and action (actor) approximation for CACLA learner"""
    def __init__(self, indim=1,  outdim=1):
        Module.__init__(self, indim, outdim)
        # V approximation
        self.V = np.zeros((indim, 1))
        self.Vactf = lambda x: x
        self.Vw = np.random.uniform(-1e-5, 1e-5, self.V.T.shape)
        self.Vcurr = 0.
        self.Vlast = 0.
        # A approximation
        self.A = np.zeros((indim, 1))
        self.Aactf = lambda x: x
        self.Aw = np.random.uniform(-1e-5, 1e-5, self.A.T.shape)
        self.Acurr = 0.
        self.Alast = 0.

    def getValue(self, state):
        return np.dot(self.Vw, state)

    def _forwardImplementation(self, inbuf, outbuf):
        # print "aclayer.py:inbuf", inbuf
        # outbuf[:] = np.tanh(np.dot(self.Aw, np.asarray(inbuf)))
        outbuf[:] = np.dot(self.Aw, np.asarray(inbuf))
        # print "aclayer.py:outbuf", outbuf
        # outbuf[:] = inbuf

    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        inerr[:] = outerr
