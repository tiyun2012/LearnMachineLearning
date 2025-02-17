import numpy
# scipy.special for the sigmoid function expit()
import scipy.special
# library for plotting arrays
import matplotlib.pyplot
import json, os

ROOT = os.path.dirname(os.path.realpath(__file__))
# ROOT = "/content/sample_data"
DATADIR = "%s/temp/data"%ROOT

# neural network class definition
class neuralNetwork:

    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # link weight matrices, wih and who
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        # learning rate
        self.lr = learningrate

        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)

        pass


    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))

        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))

        pass

    def saveW(self):
        numpy.save( "/content/sample_data/who.npy", self.who)

    def loadW(self):
        self.who = numpy.load( "/content/sample_data/who.npy")

    # query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

# number of input, hidden and output nodes
input_nodes = 12
hidden_nodes = 200
output_nodes = 200

# learning rate
learning_rate = 0.1

# create instance of neural network
n = neuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)

# load data
# Open and read the JSON file
with open('%s/animationData.json'%ROOT, 'r') as file:
    dataIn = json.load(file)
with open('%s/jacket_dembone_data.json'%ROOT, 'r') as file:
    dataOut = json.load(file)
_ids = []
_inputs = []
_outputs = []
for k, v in dataIn.items():
    _ids.append(k)
    tmp = []
    tmp.extend([i for i in v["clavicle_l"]])
    tmp.extend([i for i in v["lowerarm_l"]])
    _inputs.append(tmp)
for k, v in dataOut.items():
    tmp = []
    tmp.extend([i for i in v["clavicle"]])
    tmp.extend([i for i in v["hand"]])
    tmp.extend([i for i in v["root"]])
    tmp.extend([i for i in v["sub_clavicle"]])
    _outputs.append(tmp)

_inputs = numpy.array(_inputs, dtype='f').reshape(200,12)
_outputs = numpy.array(_ids,dtype='f').reshape(200,1)

# print(_outputs[0])
# print(_inputs[0])

# train the neural network

# epochs is the number of times the training data set is used for training
epochs = 25

for e in range(epochs):
    # go through all records in the training data set
    i = 0
    for x, y in zip(_inputs,_outputs):
      inputs = x
      # targets = y
      targets = numpy.zeros(output_nodes) + 0.01
      targets[i] = 0.99
      n.train(inputs, targets)
      i+=1