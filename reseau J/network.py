from relu import *
from tanh import *
from sigmoid import *

from layer import *

class Network():

    def __init__(self):
        self.layers = []

        i = Layer(ReLu, 4)

        f1 = Layer(ReLu, 3)
        f2 = Layer(ReLu, 3)

        h1 = Layer(ReLu, 2)
        h2 = Layer(ReLu, 2)
        h3 = Layer(Sigmoid, 2)

        o = Layer(ReLu, 3)


        i.convolution(f1, 2)
        i.convolution(f2, 2)

        f1.fully_connected(h1)
        f2.fully_connected(h1)

        h1.fully_connected(h2)
        h2.fully_connected(h3)
        h3.fully_connected(o)


        self.layers += [i, f1, f2, h1, h2, h3, o]

    def learn(self):
        self.forward()
        self.update_error()
        self.backward()
        self.adjust()


    def forward(self):
        for i in range(len(self.layers)) :
            for curr in self.layers[i].neurons :
                value = 0.0

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
