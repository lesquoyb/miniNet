from neuron import *


class TanH(Neuron):

    def __init__(self):
        pass

    def fn(self, x):
        return (1 - e**(-2 * x)) / (1 + e**(-2 * x))

    def der(self, x):
        return (4 * e**(2*x)) / ((1 + e**(2*x)) * (1+e**(2*x)))
