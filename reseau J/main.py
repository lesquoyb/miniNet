import numpy as np

from network import *
indices = {"Iris-setosa" : 0,
           "Iris-versicolor": 1,
           "Iris-virginica": 2}
def parseFile(filename):
    data = []
    classes = []
    for line in open(filename).read().split("\n"):
        data += [line.split(",")[:4]]
        d = [0]*3
        d[indices[line.split(",")[-1]]] = 1
        classes += [d]
    return data, classes


def main():
    data, classes = parseFile("iris.data.txt")
    nBlocks = 5
    size = len(data)//nBlocks
    bData = []
    bClasses = []
    for i in range(nBlocks):
        bData += [data[i*size: (i+1) * size]]
        bClasses += [classes[i*size: (i+1) * size]]

    network = Network()

    network.learn(data, classes, 5) # TEMPORAIRE, car que 4 entree dans l'appel en dessous

    error = 0
    for i in range(nBlocks):
        tData = bData[i]
        tClasses = bClasses[i]
        data = [ b for j,b in enumerate(bData) if j != i]
        classes = []
        for j,b in enumerate(bClasses):
            if j != i:
                classes += b
        network.learn(data, classes, 5)
        error += network.error_rate(tData, tClasses)
    print(str(error/nBlocks) + "% d'erreur")


if __name__ == "__main__":
    main()
