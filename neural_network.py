import sys
import numpy
import scipy.special
import matplotlib.pyplot

class neural_network:

    def __init__(self, network_list, learningrate):

        self.network = network_list
        self.network_layers = len(self.network)

        if (self.network_layers < 3):
            print('least layers is 3')
            return -1

        self.input_nodes = self.network[0]
        self.output_nodes = self.network[-1]

        # calc layers include hidden layers and output layers
        self.calc_layers = self.network_layers - 1
        self.calc_inputs = [None] * self.calc_layers
        self.calc_outputs = [None] * self.calc_layers
        self.calc_errors = [None] * self.calc_layers

        self.w_number = self.network_layers - 1
        self.w = [None] * self.w_number
        for i in range(self.w_number):
            self.w[i] = numpy.random.normal(0.0, pow(self.network[i], -0.5), (self.network[i+1], self.network[i]))

        self.lr = learningrate

        self.activation_function = lambda x: scipy.special.expit(x)

        print("initial network is " + str(self.network))
        pass

    def train(self, inputs_list, targets_list):

        if (len(inputs_list) != self.input_nodes):
            print("error : input nodes shoule be " + str(self.input_nodes))
            return -1
        if (len(targets_list) != self.output_nodes):
            print("error : output nodes shoule be " + str(self.output_nodes))
            return -1

        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        temp = inputs.copy()
        for i in range(self.calc_layers):
            self.calc_inputs[i] = numpy.dot(self.w[i], temp)
            self.calc_outputs[i] = self.activation_function(self.calc_inputs[i])
            temp = self.calc_outputs[i]

        for i in reversed(range(self.calc_layers)):
            if (i == (self.calc_layers - 1)):
                self.calc_errors[i] = targets - self.calc_outputs[i]
            else:
                self.calc_errors[i] = numpy.dot(self.w[i+1].T, self.calc_errors[i+1])

        for i in reversed(range(self.calc_layers)):
            if (i == 0):
                pre_outputs = inputs
            else:
                pre_outputs = self.calc_outputs[i-1]
            self.w[i] += self.lr * numpy.dot(self.calc_errors[i] * self.calc_outputs[i] * (1.0 - self.calc_outputs[i]), numpy.transpose(pre_outputs))

        pass

    def query(self, inputs_list):

        if (len(inputs_list) != self.input_nodes):
            print("error : input nodes shoule be " + str(self.input_nodes))
            return -1

        value = numpy.array(inputs_list, ndmin=2).T

        for i in range(self.w_number):
            value = self.activation_function(numpy.dot(self.w[i], value))

        return value
