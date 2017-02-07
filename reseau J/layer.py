class Layer():

    def __init__(self, nType, number):
        self.neurons = []

        for i in range(number):
            self.neurons += [nType()]


    def fully_connected(layer1, layer2) :
        for n1 in layer1.neurons :
            for n2 in layer2.neurons :
                n1.addSuccessor(n2)


    def convolution(layer1, layer2, asso) :
        for i in range(len(layer2.neurons)) :
            for j in range(asso) :
                layer1.neurons[i + j].addSuccessor(layer2.neurons[i])
