from neuron import Neuron
from math import e

class ReLu(Neuron):


    def __init__(self):
        pass


    def fn(self, x):
        return max(x, 0)

    def der(self, x):
        return 1.0/(1 + e**-x)


