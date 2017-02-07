from neuron import *
from relu import *
from tanh import *
from sigmoid import *

from layer import *

class Network():

    def __init__(self):
        i = Layer(Neuron, 4)

        f1 = Layer(ReLu, 3)
        f2 = Layer(ReLu, 3)

        h1 = Layer(ReLu, 2)
        h2 = Layer(ReLu, 2)
        h3 = Layer(TanH, 2)

        o = Layer(ReLu, 3)



        i.convolution(f1, 2)
        i.convolution(f2, 2)

        f1.fully_connected(h1)
        f2.fully_connected(h1)

        h1.fully_connected(h2)
        h2.fully_connected(h3)
        h3.fully_connected(o)

        i.fully_connected(h2)
        i.fully_connected(h3)

        self.layers = [i, f1, f2, h1, h2, h3, o]
        self.hiddenLayers = [f1, f2, h1, h2, h3]
        self.inputLayer = i
        self.outputLayer = o

    def error_rate(self, data, output):
        sum = 0
        nbData = len(data)
        for i in range(nbData):
            self.forward(data[i])
            for j in range(len(self.outputLayer.neurons)) :
                nodeValue = int(self.outputLayer.neurons[j].value)
                expectedValue = output[i][j]
                if  nodeValue != expectedValue :
                    sum += 1
                    break
        return (sum / nbData) * 100

    def learn(self, inputs, outputs, criterion):
        error_rate = 100;
        while error_rate > criterion :
            err = 0
            for i in range(len(inputs)) :
                self.forward(inputs[i])
                err += self.update_error(outputs[i])
                self.backward()
                self.adjust()
            error_rate = self.error_rate(inputs, outputs)
            print ( "learning : " + str(error_rate) + "% d'erreur   " + str(err) )

    def forward(self, inputs):
        for layer in self.layers :
            if layer == self.inputLayer :
                for i in range(len(layer.neurons)) :
                    layer.neurons[i].value = float(inputs[i])
            else :
                for neuron in layer.neurons :
                    neuron.value = 0.

        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.value = neuron.fn(neuron.value)
                for n, w in neuron.outputs:
                    n.value += neuron.value * w
    #TODO remetre value à 0


    def update_error(self, expected):
        total_error = 0.0
        for i in range(len(self.outputLayer.neurons)) :
            out = self.outputLayer.neurons[i]
            err = expected[i] - self.outputLayer.neurons[i].value
            out.error = err * out.der(out.value)
            total_error += 0.5 * err * err
        return total_error


    def backward(self):
        for layer in self.hiddenLayers[-1:0:-1] :
            for curr in layer.neurons :
                error = 0
                for (succ, weight) in curr.outputs:
                    error += succ.error * weight
                curr.error = error * curr.der(curr.value)

    def adjust(self):
        for layer in self.hiddenLayers :
            for curr in layer.neurons :
                nOutputs = []
                for (succ, weight) in curr.outputs :
                    weight += succ.error * curr.value
                    nOutputs += [(succ, weight)]
                curr.outputs = nOutputs


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
