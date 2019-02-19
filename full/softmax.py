import numpy as np


class SoftMaxLayer(object):
    def __init__(self):
        """
        Constructor
        """
        self.y = None

    def forward(self, x):
        """
        Implement forward pass of softmax
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
             The output of the layer (needed for backpropagation)
        """
        temp_max = np.max(x)
        temp = x - temp_max
        temp_exp = np.exp(temp)
        temp_sum = np.sum(temp_exp,axis = 1).reshape(np.shape(x)[0],1)
        self.y = temp_exp/(temp_sum)
        return self.y

    def backward(self, y_grad):
        """
        Compute "backward" computation of softmax
        Parameters
        ----------
        y_grad : np.array
            The gradient at the output
        Returns
        -------
        np.array
            The gradient at the input
        """
        grad = np.ones((np.shape(self.y)[0],np.shape(self.y)[1]))
        for i in range(np.shape(self.y)[0]):
            z = np.reshape(self.y[i,:],(1, np.shape(self.y)[1]))
            diag = np.diag(self.y[i,:])
            grad_z = diag - np.dot(z.T,z)
            y = np.reshape(y_grad[i,:],(1,np.shape(y_grad)[1]))

            out = np.dot(y,grad_z)

            # grad_z = (np.diag(self.y[i, :]) - np.dot(self.y[i, :].T, self.y[i, :]))
            grad[i,:] = out
        self.grad = grad
        # print('soft',grad)
        return grad


    def update_param(self, lr):
        pass  # no learning for softmax layer