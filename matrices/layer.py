import numpy as np
class Layer():


    def __init__(self, data_in,  fn, outputs):
        self.inputs = np.array(data_in)
        self.outputs = np.array(outputs)
        self.weights = 2 * np.random.random((len(data_in[0]), 1))-1
        self.function = fn
        self.error = 0


    def forward(self):

        res = self.function.fn(np.dot(self.inputs, self.weights))
        error = res - self.outputs
        self.weights += np.dot(self.inputs.T, error * self.function.der(res))
        self.error = error.sum()
