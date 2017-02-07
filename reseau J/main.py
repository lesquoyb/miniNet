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
    print(data)
    print(classes)
    nBlocks = 5
    size = len(data)//nBlocks
    bData = []
    bClasses = []
    for i in range(nBlocks):
        bData += [data[i*size: (i+1) * size]]
        bClasses += [classes[i*size: (i+1) * size]]

    network = Network()

    error = 0
    for i in range(nBlocks):
        tData = bData[i]
        tClasses = bClasses[i]
        data = [ b for j,b in enumerate(bData) if j != i]
        classes = [ b for j,b in enumerate(bClasses) if j != i]
        network.learn(data, classes)
        error += network.error_rate(tData, tClasses)
    print(str(error/(len(data)*nBlocks) + "% d'erreur"))


if __name__ == "__main__":
    main()
