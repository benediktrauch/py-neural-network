import random
import math
import tools
import matrix

# Loading import
import itertools
import threading
import sys

# Plot import
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Learning animation
# thanks to Andrew Clark on
# https://stackoverflow.com/questions/22029562/python-how-to-make-simple-animated-loading-while-process-is-running
def animate():
    for c in itertools.cycle(['.', '..', '...', '   ']):  # ['|', '/', '-', '\\']
        if done:
            break
        sys.stdout.write("\rI'm  learning " + c)
        sys.stdout.flush()
        time.sleep(0.5)
    sys.stdout.write('\rDone!                               ')


# My net
def main(x, hidden, b, learning, test, w, e, n_d):
    random.seed(1)
    training_data = x[:]
    noise_data = n_d[:]

    # Adding bias to training data
    training_data.append([])
    noise_data.append([])
    for _ in x[0]:
        training_data[1].append(b)
        noise_data[1].append(b)

    # Random weights for synapses
    synapses0 = []
    synapses1 = []

    for _ in range(hidden):
        synapses0.append([random.uniform(w, -w), random.uniform(w, -w)])  # second rand for bias synapses
    for j in range(hidden + 1):  # +1 for bias
        synapses1.append([random.uniform(w, -w)])

    sig_layer2 = []

    # learning loop (learning = iterations)
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

        # if i % 25000 == 0:
        #     temp = 0
        #     for j in range(len(error)):
        #         temp += temp + error[0][j]
        #     print i, temp

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

    # Testing net with noised data
    sig_noise = []
    if len(n_d) > 0:
        # print "testing with noise data"

        l1 = matrix.multiply(synapses0, noise_data)

        sig_l1 = matrix.sig(l1)

        b_sig_l1 = sig_l1[:]

        b_sig_l1.append([])

        for _ in b_sig_l1[0]:
            b_sig_l1[len(b_sig_l1) - 1].append(b)

        l2 = matrix.multiply(matrix.transpose(synapses1), b_sig_l1)

        sig_noise = matrix.sig(l2)

    # formatting net output for plot
    result1 = []  # training data
    result2 = []  # noised data
    for i in range(len(sig_layer2[0])):
        result1.append(sig_layer2[0][i] * 2 - 1)
        result2.append(sig_noise[0][i] * 2 - 1)

    # Plot
    # Some code lines from: https://matplotlib.org/users/legend_guide.html
    neuron_patch = mpatches.Patch(label='Neurons: '+str(hidden))
    bias_patch = mpatches.Patch(label='Bias: '+str(b))
    iteration_patch = mpatches.Patch(label='Iterations: '+str(learning))
    epsilon_patch = mpatches.Patch(label='Epsilon: '+str(e))
    weight_patch = mpatches.Patch(label='Weight range (0 +/-): '+str(w))
    time_patch = mpatches.Patch(label=str(round((time.time() - start_time)/60, 2)) + " min")
    first_legend = plt.legend(handles=[bias_patch, time_patch, epsilon_patch, neuron_patch, iteration_patch, weight_patch],
                              bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                              ncol=3, mode="expand", borderaxespad=0.)

    line1, = plt.plot(inputData[0], result1, label="Training Data", linewidth=0.75)
    line2, = plt.plot(inputData[0], result2, label="Test Data", linestyle=':', linewidth=0.75)
    line3, = plt.plot(x_data, y_data, label="sin(x)", linestyle='--', linewidth=0.75)
    ax = plt.gca().add_artist(first_legend)
    plt.legend(handles=[line1, line2, line3])
    plt.savefig('./plots/plot'+str(time.time())+'.png')

    plt.clf()
    plt.cla()
    plt.close()


# inputData = [[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]]


inputData = [tools.linspace(0, 6.4, 50)]  # 2 * math.pi

testdata = []
for data in range(len(inputData[0])):
    testdata.append([(math.sin(inputData[0][data])*0.5)+0.5])

noise_d = [tools.linspace(0.1, 6.5, 50)]

x_data = inputData[0]
y_data = np.sin(x_data)

iterations = 50000
hiddenNeurons = 13
bias = 1.
weight = 0.95
epsilon = 0.41

for _ in range(1):
    done = False

    t = threading.Thread(target=animate)
    t.start()
    # epsilon += 0.01
    # print "epsilon : " + str(epsilon)
    start_time = time.time()

    main(inputData, hiddenNeurons, bias, iterations, testdata, weight, epsilon, noise_d)

    time.sleep(10)
    done = True





