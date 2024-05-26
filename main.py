import numpy as np
import matplotlib.pyplot as plt

from layer import *
from activation import *
from loss import *
from data import *
from optimisers import *

np.random.seed(0)

# Define data
X, y = spiral_data(100,3)
#plt.scatter(X[:,0], X[:,1], c=y, s=40, cmap='viridis')
#plt.show()

# Initialise neural net
dense1 = Layer_Dense(2,64)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64,3)
loss_activation = Loss_CatCrossEntropy_with_SoftMax()
optimiser = Optimiser_SGD(learning_rate=1, decay=1e-3)

# Initialise serialisation
training_data = []

for epoch in range(20001):
    # Feedforward data through network
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.calculate(dense2.output, y)

    # Compute accuracy
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions==y)

    # Save/print training data for current epoch
    if not epoch % 100:
        '''print(f'epoch: {epoch}, ' +
            f'acc: {accuracy:.3f}, ' +
            f'loss: {loss:.3f}, ' +
            f'lrate: {optimiser.current_learning_rate:.3f}')'''
        training_data.append([epoch, accuracy, loss, optimiser.current_learning_rate])

    # Save training data
    

    # Backpropagate derivatives
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Perform gradient descent step
    optimiser.pre_update()
    optimiser.update_params(dense1)
    optimiser.update_params(dense2)
    optimiser.post_update()

# Serialise training data
training_data_np = np.array(training_data)
np.savetxt("training_data.csv", training_data_np, delimiter=",")

plt.plot(training_data_np[:,0], training_data_np[:,1])
plt.plot(training_data_np[:,0], training_data_np[:,2])
plt.show()