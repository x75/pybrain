__author__ = 'Oswald Berthold'

from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import TanhLayer, LinearLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
# from pybrain.structure.modules.linearlayer import LinearLayer
from pybrain.structure.modules.module import Module

import numpy as np

# FIXME: make V and A use the same expanded state (Astate = Vstate)

class ACLayer(Module):
    """Module combining value (critic) and action (actor) approximation for CACLA learner"""
    def __init__(self, indim=1,  outdim=1, hdim=1):
        Module.__init__(self, indim, outdim)
        self.hdim = hdim
        self.indim = indim
        self.outdim = outdim
        self.winamp = 1.
        self.wactamp = 1.
        # V approximation
        self.V = np.zeros((hdim, 1))
        self.Vactf = lambda x: x
        # self.Vwin = np.random.uniform(-1, 1, (self.hdim, indim)) * self.winamp
        self.Vwin = np.random.normal(0, 1., (self.hdim, indim)) * self.winamp
        # self.Vw   = np.random.uniform(-1e-5, 1e-5, self.V.T.shape)
        self.Vw   = np.zeros(self.V.T.shape)
        self.Vcurr = 0.
        self.Vlast = 0.
        self.Vstate = np.zeros((self.hdim, 1))
        # A approximation
        self.A = np.zeros((hdim, 1))
        self.Aactf = lambda x: x
        # self.Awin = np.random.uniform(-1, 1, (self.hdim, indim)) * self.wactamp
        self.Awin = np.random.normal(0., 1, (self.hdim, indim)) * self.wactamp
        # self.Aw   = np.random.uniform(-1e-5, 1e-5, self.A.T.shape)
        print "outdim", outdim
        self.Aw   = np.zeros((outdim, hdim)) # rows = number of output units
        self.Acurr = 0.
        self.Alast = 0.
        self.Astate = np.zeros((self.hdim, 1))

    def trainV(self, beta, target, target_, state):
        # determine delta
        delta = (target - target_) * self.Vstate
        # move weights by beta * delta
        self.Vw += beta * delta

    def trainA(self, alpha, target, target_, state):
        # lastaction = self.getAction(state)
        target = target.reshape((self.outdim,1))
        target_ = target_.reshape((self.outdim,1))
        d1 = (target - target_)
        # print d1.shape
        d2 = d1 * self.Astate
        # print d2.shape
        delta = d2.reshape((self.outdim, self.hdim))
        # print "Atarget.shape", Atarget.shape, self.module.Aw.shape
        # Atst = np.dot(self.module.Aw, self.laststate)
        # delta = target - Atst
        # print "target", target
        self.Aw += alpha * delta

    def getValue(self, state):
        self.Vstate = np.tanh(np.dot(self.Vwin, state))
        self.Vstate[-1] = 1.
        self.Vstate += np.random.normal(0., 0.01, self.Vstate.shape)
        # print "y.shape", y.shape
        return np.dot(self.Vw, self.Vstate)
        # return np.dot(self.Vw, state)
        
    def getAction(self, state):
        self.Astate = np.tanh(np.dot(self.Awin, state))
        self.Astate[-1] = 1.
        self.Astate += np.random.normal(0., 0.01, self.Astate.shape)
        # print "y.shape", y.shape
        return np.dot(self.Aw, self.Astate)
        # return np.dot(self.Vw, state)

    def _forwardImplementation(self, inbuf, outbuf):
        # print "aclayer:forwardimpl", inbuf
        # print "aclayer.py:inbuf", inbuf
        # outbuf[:] = np.tanh(np.dot(self.Aw, np.asarray(inbuf)))
        
        self.Astate = np.tanh(np.dot(self.Awin, np.asarray(inbuf))).reshape((self.hdim, 1))
        # self.Astate = 0.99 * self.Astate + 0.01 * nstate
        self.Astate[-1] = 1.
        self.Astate += np.random.normal(0., 0.01, self.Astate.shape)
        # print "y.shape", y.shape
        outbuf[:] = np.dot(self.Aw, self.Astate).reshape((self.outdim))
        # outbuf[:] = np.dot(self.Aw, np.asarray(inbuf))
        # print "aclayer.py:outbuf", self.Astate
        # print "aclayer.py:outbuf", outbuf
        # outbuf[:] = inbuf

    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        inerr[:] = outerr


class ACLayerMLP(Module):
    """Module combining value (critic) and action (actor) approximation for CACLA learner, std MLP variant"""
    def __init__(self, indim=1,  outdim=1, hdim=1):
        Module.__init__(self, indim, outdim)
        self.hdim = hdim
        self.indim = indim
        self.outdim = outdim

        self.Vnet = buildNetwork(self.indim, self.hdim, 1, hiddenclass=TanhLayer, outclass=LinearLayer, bias=True)
        self.Vds = SupervisedDataSet(self.indim, 1)
        self.Anet = buildNetwork(self.indim, self.hdim, self.outdim, hiddenclass=TanhLayer, outclass=LinearLayer, bias=True)
        self.Ads = SupervisedDataSet(self.indim, self.outdim)

    def trainV(self, beta, target, target_, state):
        self.Vds.addSample(state, target)
        trainer = BackpropTrainer(self.Vnet, self.Vds, momentum=0.0, verbose=True)
        trainer.train()

    def trainA(self, alpha, target, target_, state):
        self.Ads.addSample(state, target)
        trainer = BackpropTrainer(self.Anet, self.Ads, momentum=0.0, verbose=True)
        trainer.train()

    def getValue(self, state):
        return self.Vnet.activate(state)
        
    def getAction(self, state):
        return self.Anet.activate(state)

    def _forwardImplementation(self, inbuf, outbuf):
        outbuf[:] = self.Anet.activate(inbuf)

    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        inerr[:] = outerr
