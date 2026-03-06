
import numpy as np

class NeuralLayer:
    def __init__(self, weight_init_method, input_nodes, output_nodes):
        #Initializing weights based on weight method
        if weight_init_method == "random":
            self.W = np.random.randn(input_nodes, output_nodes) * 0.01
        elif weight_init_method == "xavier":
            scaler = (2 / (input_nodes + output_nodes))**(0.5)
            self.W = np.random.randn(input_nodes, output_nodes) * scaler

        #Initializing grads and b values to zeros    
        self.b = np.zeros(output_nodes) 
        self.grad_W = np.zeros((input_nodes, output_nodes))
        self.grad_b = np.zeros(output_nodes)
        self.X = None

    def forward(self, X):
        self.X = X
        #Forward pass formula.
        Z = X @ self.W + self.b
        return Z

    def backward(self, grad_output):
        #Backward pass maths
        self.grad_W = (self.X).T @ grad_output
        self.grad_b = np.sum(grad_output, axis=0) 
        grad_input =  grad_output @ (self.W).T
        return grad_input