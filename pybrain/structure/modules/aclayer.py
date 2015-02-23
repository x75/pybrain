__author__ = 'Oswald Berthold'

from pybrain.structure.modules.linearlayer import LinearLayer
from pybrain.structure.modules.module import Module

import numpy as np


class ACLayer(Module):
    """Module combining value (critic) and action (actor) approximation for CACLA learner"""
    def __init__(self, indim=1,  outdim=1, hdim=1):
        Module.__init__(self, indim, outdim)
        self.hdim = hdim
        self.winamp = 0.1
        self.wactamp = 0.1
        # V approximation
        self.V = np.zeros((hdim, 1))
        self.Vactf = lambda x: x
        self.Vwin = np.random.uniform(-1, 1, (self.hdim, indim)) * self.winamp
        # self.Vw   = np.random.uniform(-1e-5, 1e-5, self.V.T.shape)
        self.Vw   = np.zeros(self.V.T.shape)
        self.Vcurr = 0.
        self.Vlast = 0.
        # A approximation
        self.A = np.zeros((hdim, 1))
        self.Aactf = lambda x: x
        self.Awin = np.random.uniform(-1, 1, (self.hdim, indim)) * self.wactamp
        # self.Aw   = np.random.uniform(-1e-5, 1e-5, self.A.T.shape)
        self.Aw   = np.zeros(self.A.T.shape)
        self.Acurr = 0.
        self.Alast = 0.

    def getValue(self, state):
        y = np.tanh(np.dot(self.Vwin, state))
        # print "y.shape", y.shape
        return np.dot(self.Vw, y)
        # return np.dot(self.Vw, state)
        
    def getAction(self, state):
        y = np.tanh(np.dot(self.Awin, state))
        # print "y.shape", y.shape
        return np.dot(self.Aw, y)
        # return np.dot(self.Vw, state)

    def _forwardImplementation(self, inbuf, outbuf):
        # print "aclayer.py:inbuf", inbuf
        # outbuf[:] = np.tanh(np.dot(self.Aw, np.asarray(inbuf)))
        y = np.tanh(np.dot(self.Awin, np.asarray(inbuf))).reshape((self.hdim, 1))
        # print "y.shape", y.shape
        outbuf[:] = np.dot(self.Aw, y)
        # outbuf[:] = np.dot(self.Aw, np.asarray(inbuf))
        # print "aclayer.py:outbuf", outbuf
        # outbuf[:] = inbuf

    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        inerr[:] = outerr
