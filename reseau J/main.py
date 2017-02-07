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
    network = Network()
    network.learn()


if __name__ == "__main__":
    main()
