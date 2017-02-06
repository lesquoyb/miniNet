from relu import *
from tanh import *
from sigmoid import *

from layer import *

class Network():

    def __init__(self):
        self.layers = []

        h1 = Layer(ReLu, 2)
        h2 = Layer(ReLu, 2)
        h3 = Layer(Sigmoid, 2)

        h1.fully_connected(h2)
        h2.fully_connected(h3)

        self.layers += [h1, h2, h3]

        print (self.layers)

    def learn(self):
        self.forward()
        self.update_error()
        self.backward()
        self.adjust()


    def forward(self):
        for i in range(len(self.layers)) :
            for curr in self.layers[i].neurons :
                value = 0.0

                print (curr)

        pass

    def update_error(self):
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
