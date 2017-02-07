from neuron import *


class TanH(Neuron):

    def fn(self, x):
        #return (1 - e**(-2 * x)) / (1 + e**(-2 * x))
        return (e**x - e**-x) / (e**x + e**-x)

    def der(self, x):
        #return (4 * e**(2*x)) / ((1 + e**(2*x)) * (1+e**(2*x)))
        return 1 - self.fn(x)**2
