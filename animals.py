import neural_network as nn
import numpy
import sys
import cv2
import os
import random

base_dir = '/home/neo/disk/ml/book_n/neural_network_python/animals/'
categories = ['dogs', 'cats', 'panda']
print_l = ['-', '/', '|', '\\']

HEIGHT = 32
WIDTH = 55
scorecard = []
data_sets = []

input_nodes = HEIGHT * WIDTH
output_nodes = len(categories)

n = nn.neural_network([input_nodes, 700, output_nodes], 0.1)

print('Get datasets...')

test_num=0
for k, category in enumerate(categories):
    for f in os.listdir(base_dir + category):
        test_num += 1
        image = cv2.imread(base_dir + category + '/' + f)
        image = cv2.resize(image, (WIDTH, HEIGHT))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.flatten()
        inputs = (image / 255.0 * 0.99) + 0.01
        targets = numpy.zeros(output_nodes) + 0.01
        targets[k] = 0.99
        data_sets.append({'inputs':inputs, 'results':targets})
        pass
    pass

print('Begin training...')

random.shuffle(data_sets)

test_idx=0
for tests in data_sets:
    test_idx += 1
    print(str(test_idx*100.0/test_num) + '%', end='\r')
    n.train(tests['inputs'], tests['results'])
    pass
pass

print('Begin verify...')
print(str(categories))

for f in os.listdir(base_dir + 'images'):
    print('.', end=' ')
    image = cv2.imread(base_dir + 'images/' + f)
    image = cv2.resize(image, (WIDTH, HEIGHT))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.flatten()
    inputs = (image / 255.0 * 0.99) + 0.01
    outputs = n.query(inputs)
    label = numpy.argmax(outputs)
    
    if f.split('.')[0][0:3] in categories[label]:
        scorecard.append(1)
    else:
        scorecard.append(0)

    print(f + ' is ' + str(outputs.T))
    pass

scorecard_array = numpy.asarray(scorecard)
print ("performance = %", scorecard_array.sum() * 100.0 / scorecard_array.size)
