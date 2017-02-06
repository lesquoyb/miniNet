import numpy as np

from network import *

import matplotlib.pyplot as plt


def main():
    np.random.seed(2)
    n = Network()
    n.learn()


def drawShit():
    fig, ax = plt.subplots()
    x = []
    y2 = []
    y = []
    f = Sigmoid()
    for i in range(-20, 20):
        x += [i]
        y += [f.fn(i)]
        y2 += [f.der(i)]
        print("f(" + str(i) + ")=" + str(f.fn(i)))

    scat = ax.scatter(x, y)
    ax.scatter(x, y2)
    # fig.colorbar(scat)

    plt.show()


if __name__ == "__main__":
    main()