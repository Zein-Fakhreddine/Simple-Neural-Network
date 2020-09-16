import numpy as np

class NeuralNetwork:
    
    def __init__(self, input_num, hidden_num, output_num):
        self.input_num = input_num
        self.hidden_num = hidden_num
        self.output_num = output_num
        self.input_hidden_weights = np.random.rand(hidden_num, input_num)
        self.hidden_output_weights = np.random.rand(output_num, hidden_num)
        self.hidden_bias = np.random.rand(hidden_num, 1)
        self.output_bias = np.random.rand(output_num, 1)
        self.learning_rate = 0.1
        self.activationFunctions = (NeuralNetwork.sigmoid, NeuralNetwork.sigmoid_prime)

    
    def sigmoid(num):
        return 1 / (1 + np.exp(-num))

    def sigmoid_prime(num):
        return NeuralNetwork.sigmoid(num) * (1 - NeuralNetwork.sigmoid(num))

    def relu(arr):
        return np.array([max(0, num) for num in arr]).reshape(arr.shape)

    def relu_prime(arr):
        return np.array([1 if num > 0 else 0 for num in arr]).reshape(arr.shape)

    def leaky_relu(arr):
        alpha = 0.01
        return np.array([max(alpha * num, num) for num in arr]).reshape(arr.shape)
    
    def leaky_relu_prime(arr):
        alpha = 0.01
        return np.array([1 if num > 0 else alpha for num in arr]).reshape(arr.shape)

    def elu(arr):
        alpha = 1.0
        return np.array([num if num >= 0 else alpha*(np.exp(num) -1) for num in arr]).reshape(arr.shape)

    def elu_prime(arr):
        alpha = 1.0
        return np.array([1 if num > 0 else alpha*np.exp(num) for num in arr]).reshape(arr.shape)

    def tanh(num):
	    return (np.exp(num) - np.exp(-num)) / (np.exp(num) + np.exp(-num))

    def tanh_prime(num):
        return 1 - np.power(NeuralNetwork.tanh(num), 2)

    def predict(self, inputs):
        input_array = np.array(inputs).reshape(len(inputs), 1)

        hidden = np.matmul(self.input_hidden_weights, input_array) + self.hidden_bias
        hidden_sig = self.activationFunctions[0](hidden)

        output = np.matmul(self.hidden_output_weights, hidden_sig) + self.output_bias
        output_sig = self.activationFunctions[0](output)

        return output_sig

    def train(self, inputs, targets):
        input_array = np.array(inputs).reshape(len(inputs), 1)
        answer_array = np.array(targets).reshape(len(targets), 1)

        hidden = np.matmul(self.input_hidden_weights, input_array) + self.hidden_bias
        hidden_sig = self.activationFunctions[0](hidden)
        hidden_dsig = self.activationFunctions[1](hidden)
   
        output = np.matmul(self.hidden_output_weights, hidden_sig) + self.output_bias
        output_sig = self.activationFunctions[0](output)
        output_dsig = self.activationFunctions[1](output)
        error = output_sig - answer_array
        cost = (error**2) / len(error)
        dcost = (error * 2) / len(error)

        output_gradients = output_dsig * dcost 
        hidden_output_weight_deltas = np.matmul(output_gradients, hidden_sig.T) 
        
        self.hidden_output_weights -= hidden_output_weight_deltas * self.learning_rate
        self.output_bias -= output_gradients * self.learning_rate

        hidden_gradients = np.matmul(self.hidden_output_weights.T, output_gradients) * hidden_dsig 
        input_hidden_weight_deltas = np.matmul(hidden_gradients, input_array.T)

        self.input_hidden_weights -= input_hidden_weight_deltas * self.learning_rate
        self.hidden_bias -= hidden_gradients * self.learning_rate
