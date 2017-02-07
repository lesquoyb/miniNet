from neuron import *

from numpy import *

class TanH(Neuron):

    def fn(self, x):
        #return (1 - e**(-2 * x)) / (1 + e**(-2 * x))
        return tanh(x) #(e**x - e**-x) / (e**x + e**-x)

    def der(self, x):
        #return (4 * e**(2*x)) / ((1 + e**(2*x)) * (1+e**(2*x)))
        return 1 - tanh(x)**2
