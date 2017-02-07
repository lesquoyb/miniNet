from numpy.random import random
from math import e


class Neuron:

    def __init__(self):
        self.outputs = []

        self.value = 0.0
        self.error = 0.0

    def addSuccessor(self, neuron):
        self.outputs += [(neuron, random()*0.5)]

    def fn(self, x):
        return x#à implémenter

    def der(self, x):
        return x #à implémenter

    def printer(self) :
        return str(self.value) + " err: " + str(self.error)
