from numpy.random import random
from math import e


class Neuron:

    def __init__(self):
        self.outputs = []

        self.value
        self.error

    def addSuccessor(self, neuron, weight):
        self.outputs += (neuron, weight)

    def fn(self, x):
        return 0#à implémenter

    def der(self, x):
        return 0 #à implémenter
