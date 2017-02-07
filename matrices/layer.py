import numpy as np
class Layer():


    def __init__(self, data_in,  fn, outputs):
        self.setInputs(data_in)
        self.setOutputs(outputs)
        self.weights = 2 * np.random.random((len(data_in[0]), 1))-1
        self.function = fn
        self.error = 0

    def setInputs(self, data_in):
        self.inputs = np.array(data_in)

    def setOutputs(self, outputs):
        self.outputs = np.array(outputs)


    def forward(self):

        res = self.function.fn(np.dot(self.inputs, self.weights))
        error = res - self.outputs
        self.weights += np.dot(self.inputs.T, error * self.function.der(res))
        self.error = error.sum()
