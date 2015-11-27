__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.structure.modules.neuronlayer import NeuronLayer
import numpy as np

class LinearLayer(NeuronLayer):
    """ The simplest kind of module, not doing any transformation. """

    def _forwardImplementation(self, inbuf, outbuf):
        # print("type inbuf", type(inbuf), inbuf.shape)
        outbuf[:] = inbuf + np.random.normal(0., 0.01, size=inbuf.shape)

    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        inerr[:] = outerr
