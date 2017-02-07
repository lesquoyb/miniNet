from neuron import *
from numpy import *

class Sigmoid(Neuron):

    def fn(self, x):
        if fabs(x) < 1*(10**-10) :
            return 0.5
        return 1. / (1 + e**-x)

    def der(self, x):
        #return (e**x)/( (1 + e**x) * (1 + e**x))
        return 1.0 / (1.0 + e**-x)
