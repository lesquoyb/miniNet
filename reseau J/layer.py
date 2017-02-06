class Layer():


    def __init__(self, nType, number):
        self.neurons = []

        self.neurons += [nType()]

        print (self.neurons)

    def fully_connected(layer1, layer2) :
        for n1 in layer1.neurons :
            for n2 in layer2.neurons :
                n1.addSuccessor(n2, 0.1)
