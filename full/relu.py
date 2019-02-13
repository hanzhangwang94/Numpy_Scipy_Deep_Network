import numpy as np
class ReluLayer(object):
    def __init__(self):
        """
        Rectified Linear Unit
        """
        self.y = None

    def forward(self, x):
        """
        Implement forward pass of Relu

        y = x if x > 0
        y = 0 otherwise

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number of features

        Returns
        -------
        np.array
            The output of the layer

        Stores
        -------
        self.y : np.array
             The output data (need to store for backwards pass)
        """
        y = np.copy(x)
        y[y <= 0.] = 0.0
        self.y = y
        # print('relu',y)
        return self.y

    def backward(self, y_grad):
        """
        Implement backward pass of Relu

        Parameters
        ----------
        y_grad : np.array
            The gradient at the output

        Returns
        -------
        np.array
            The gradient at the input
        """
        grad_input = np.copy(self.y)
        grad_input[grad_input > 0.] = 1.0
        grad_input = grad_input*y_grad
        self.grad_input = grad_input
        # print('relu_grad',grad_input)
        return grad_input

    def update_param(self, lr):
        pass  # no parameters to update
