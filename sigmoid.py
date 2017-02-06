from neuron import *


class Sigmoid(Neuron):

    def __init__(self):
        pass

    def fn(self, x):
        return 1. / (1 + e**-x)

    def der(self, x):
        return (e**x)/( (1 + e**x) * (1 + e**x))