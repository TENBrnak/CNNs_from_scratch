#%%
import numpy as np
import matplotlib.pyplot as plt
#%%
class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.normal(size=(n_inputs, n_neurons)) * np.sqrt(1 / n_inputs)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs


    def backward(self, d_values):
        self.dweights = np.dot(self.inputs.T, d_values)
        self.dbiases = np.sum(d_values, axis=0, keepdims=True)
        self.dinputs = np.dot(d_values, self.weights.T)
#%%
class ActivationReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        self.inputs = inputs

    def backward(self, d_values):
        self.dinputs = d_values.copy()
        self.dinputs[self.inputs <= 0] = 0
#%%
class ActivationSoftmaxLossCategoricalCrossentropy:
    def forward(self, inputs, correct_labels):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) # -np.max(...) for numerical stability with big number
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        samples = len(self.output)
        predictions = np.clip(self.output, 1e-7, 1 - 1e-7)
        correct_predictions = predictions[range(samples), correct_labels]
        per_sample_losses = -np.log(correct_predictions)
        return np.mean(per_sample_losses)

    def backward(self, dvalues, correct_labels):
        samples = len(dvalues)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), correct_labels] -= 1
        self.dinputs /= samples
#%%
class OptimizerSGD:
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate

    def update_params(self, layer):
        
        weight_updates = -self.learning_rate * layer.dweights
        bias_updates = -self.learning_rate * layer.dbiases
        
        layer.weights += weight_updates
        layer.biases += bias_updates

#%%
# read mnist data
def read_mnist_images(filename):
    with open(filename, 'rb') as f:
        data = f.read()
        assert int.from_bytes(data[:4], byteorder='big') == 2051
        n_images = int.from_bytes(data[4:8], byteorder='big')
        n_rows = int.from_bytes(data[8:12], byteorder='big')
        n_cols = int.from_bytes(data[12:16], byteorder='big')
        images = np.frombuffer(data, dtype=np.uint8, offset=16).reshape(n_images, n_rows, n_cols)
        return images, n_images

def read_mnist_labels(filename):
    with open(filename, 'rb') as f:
        data = f.read()
        assert int.from_bytes(data[:4], byteorder='big') == 2049
        labels = np.frombuffer(data, dtype=np.uint8, offset=8)
        return labels

#%%
x, num_inputs = read_mnist_images('mnist/train-images.idx3-ubyte')
y = read_mnist_labels('mnist/train-labels.idx1-ubyte')
#%%
x.shape
#%%
epochs = 100
X, Y = x / 255., y
Y = Y.reshape(-1, 1) 
X = X.reshape(-1, 784)
dense1 = LayerDense(784, 10) 
activation1 = ActivationReLU() 
dense2 = LayerDense(10, 10)
activation2 = ActivationSoftmaxLossCategoricalCrossentropy()
optimizer = OptimizerSGD()
for sample in range(60):
    for epoch in range(epochs):
        dense1.forward(X[sample*1000:(sample+1)*1000])
        activation1.forward(dense1.output)
        print((activation1.output != 0).any())
        dense2.forward(activation1.output)
        data_loss = activation2.forward(dense2.output, Y)
        predictions = np.argmax(activation2.output, axis=1)
        accuracy = np.mean(predictions == Y)
        print(f'epoch: {epoch}, ' + f'acc: {accuracy:.3f}, ' + f'loss: {data_loss:.3f}')
        activation2.backward(activation2.output, Y)
        dense2.backward(activation2.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)
        print((dense1.dweights != 0).any())
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)