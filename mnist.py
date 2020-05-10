import neural_network as nn
import numpy
import sys

input_nodes = 784
output_nodes = 10

n = nn.neural_network([input_nodes, 200, output_nodes], 0.1)

training_data_file = open("/home/neo/disk/ml/book_n/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()
training_num = len(training_data_list)

epochs = 1

training_total_num = training_num * epochs
training_idx = 0

print('Begin training...')

for e in range(epochs):
    for record in training_data_list:
        sys.stdout.write('\r%%%8f\t' %(100.0 * training_idx / training_total_num))
        training_idx += 1
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    pass

test_data_file = open("/home/neo/disk/ml/book_n/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

print('Begin verify...')

scorecard = []

for record in test_data_list:
    all_values = record.split(',')
    correct_label = int(all_values[0])
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    outputs = n.query(inputs)
    label = numpy.argmax(outputs)
    if (label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    
    pass
	
scorecard_array = numpy.asarray(scorecard)
print ("performance = %", scorecard_array.sum() * 100.0 / scorecard_array.size)
