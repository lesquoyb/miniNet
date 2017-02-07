from neuron import *


class Sigmoid(Neuron):

    def fn(self, x):
        return 1. / (1 + e**-x)

    def der(self, x):
        return (e**x)/( (1 + e**x) * (1 + e**x))
