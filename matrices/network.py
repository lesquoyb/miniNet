from relu import *
from tanh import *
from sigmoid import *
from sigmoid import Sigmoid
from layer import Layer
import numpy

class Network():

    def __init__(self):
        data = [[0.5], [0.34], [0.14], [0.02]]
        self.layers = [Layer(data, fn=ReLu(), outputs=numpy.array([[1],[0],[0]]).T)]



    def learn(self):
        epsilon = 0.1
        error = 10 * epsilon
        while abs(error) > epsilon:
            error = 0
            for layer in self.layers:
                layer.forward()
                error += layer.error
            error /= len(self.layers)
            print(error)

        print(self.layers[0].weights)


    def forward(self):
        pass

    def update_erro(self):
        pass


    def backward(self):
        pass

    def adjust(self):
        pass
'''
FORWARD :
    pour chaque couche :
        pour chaque neurone de la couche (curr) :
            value = 0;

            pour chaque neurones antécédents (pred) :
                value += pred.weight * pred.output

            curr.output = layer.fn(value)

TOTAL_ERROR :
    pour chaque neurone de sortie (out) :
        err = expected[i] - out[i].output
        out.error = err * der(out[i].output)
        total_error += 0.5  err  err  # Erreur quadratique

    return total_error

BACKWARD :
    pour chaque couche cachée dans le sens inverse :
        pour chaque neurone (curr) :
            error = 0

            pour chaque neurone successeur (succ) :
                error += succ.error * curr.weight(succ)

            curr.error = error * der (curr.output)

ADJUST :
    pour chaque couche sauf sortie :
        pour chaque neurone (curr) :
            pour chaque neurone successeur (succ) :
                curr.weigth(succ) += succ.error * curr.output
'''
