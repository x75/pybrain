__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy import tanh
from numpy import random
from pybrain.structure.modules.neuronlayer import NeuronLayer


class TanhLayer(NeuronLayer):
    """ A layer implementing the tanh squashing function. """

    def _forwardImplementation(self, inbuf, outbuf):
        # outbuf[:] = tanh(inbuf)
        outbuf[:] = tanh(inbuf) + random.normal(0., 0.01, size=inbuf.shape)

    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        inerr[:] = (1 - outbuf**2) * outerr
