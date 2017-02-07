from neuron import *


class TanH(Neuron):
    
    def fn(self, x):
        return (1 - e**(-2 * x)) / (1 + e**(-2 * x))

    def der(self, x):
        return (4 * e**(2*x)) / ((1 + e**(2*x)) * (1+e**(2*x)))
