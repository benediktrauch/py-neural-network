import random
import math
import tools
import matrix
import time

import numpy as np
import matplotlib.pyplot as plt


def main(x, hidden, b, learning, test, w, e):
    random.seed(1)
    training_data = x[:]

    # Adding bias to training data
    training_data.append([])
    for _ in x[0]:
        training_data[1].append(b)

    # Random weights for synapses
    synapses0 = []
    synapses1 = []

    for _ in range(hidden):
        synapses0.append([random.uniform(w, -w), random.uniform(w, -w)])
    for j in range(hidden + 1):  # +1 for bias
        synapses1.append([random.uniform(w, -w)])

    for i in xrange(learning):

        # # # Forward pass
        # # Input Layer

        layer1 = matrix.multiply(synapses0, training_data)

        # Activation level
        sig_layer1 = matrix.sig(layer1)

        # # Hidden Layer
        # Adding bias to layer1

        b_sig_layer1 = sig_layer1[:]

        b_sig_layer1.append([])

        for _ in b_sig_layer1[0]:
            b_sig_layer1[len(b_sig_layer1) - 1].append(b)

        layer2 = matrix.multiply(matrix.transpose(synapses1), b_sig_layer1)

        sig_layer2 = matrix.sig(layer2)

        # Calculate net error
        error = [matrix.subtract(test, matrix.transpose(sig_layer2))]

        if i % 1000 == 0:
            temp = 0
            for j in range(len(error)):
                temp += temp + error[0][j]
            print i, temp

        # Delta for neuron in output layer (1 for each training data)
        deriv_sig_layer2 = matrix.derivative(sig_layer2)
        delta_layer2 = [[]]

        for j in range(len(error[0])):
            delta_layer2[0].append(deriv_sig_layer2[0][j] * error[0][j] * e)

        # Delta for neurons in hidden layer
        deriv_sig_layer1 = matrix.derivative(sig_layer1)
        delta_layer1 = []
        delta_weight_sum = []

        for k in range(len(synapses1)):
            delta_weight_sum.append([])
            for j in range(len(delta_layer2[0])):
                delta_weight_sum[k].append(synapses1[k][0] * delta_layer2[0][j])

        for k in range(len(deriv_sig_layer1)):
            delta_layer1.append([])
            for j in range(len(deriv_sig_layer1[0])):
                delta_layer1[k].append(deriv_sig_layer1[k][j] * delta_weight_sum[k][j] * e)

        delta_w_oh = matrix.multiply(delta_layer2, matrix.transpose(b_sig_layer1))
        delta_w_hi = matrix.multiply(delta_layer1, matrix.transpose(training_data))

        # # Update weights
        synapses1 = matrix.add(synapses1, matrix.transpose(delta_w_oh))

        synapses0 = matrix.add(synapses0, delta_w_hi)

    print str(round((time.time() - start_time)/60, 2)) + " min"
    print "done"
    result = []
    for i in range(len(sig_layer2[0])):
        result.append(sig_layer2[0][i] * 2 - 1)

    # Plot
    plt.plot(inputData[0], result)
    plt.savefig('plot'+str(time.time())+'.png')
    plt.plot(inputData[0], result)
    plt.show()


# inputData = [[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]]

inputData = [tools.linspace(0., 2 * math.pi, 30)]

# print inputData

testdata = []
for data in range(len(inputData[0])):
    testdata.append([(math.sin(inputData[0][data])*0.5)+0.5])

# print testdata

x_data = inputData[0]
y_data = np.sin(x_data)

plt.plot(x_data, y_data)
# plt.savefig('plot.png')

iterations = 50000
hiddenNeurons = 15
bias = 1.
weight = 0.95
epsilon = 0.15

start_time = time.time()
main(inputData, hiddenNeurons, bias, iterations, testdata, weight, epsilon)
