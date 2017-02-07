from numpy.random import random
from math import e


class Neuron:

    def __init__(self):
        self.outputs = []

        self.value = 0.0
        self.error = 0.0

    def addSuccessor(self, neuron, weight):
        self.outputs += [(neuron, weight)]

    def fn(self, x):
        return value#à implémenter

    def der(self, x):
        return value #à implémenter
