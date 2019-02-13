import numpy as np


class MaxPoolLayer(object):
    def __init__(self, size=2):
        """
        MaxPool layer
        Ok to assume non-overlapping regions
        """
        self.locs = None  # to store max locations
        self.size = size  # size of the pooling

    def forward(self, x):
        """
        Compute "forward" computation of max pooling layer

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of input channels x number of rows x number of columns

        Returns
        -------
        np.array
            The output of the maxpooling

        Stores
        -------
        self.locs : np.array
             The locations of the maxes (needed for back propagation)
        """
        location = np.zeros_like(x)
        if np.shape(x)[2] % self.size != 0:
            sub = np.shape(x)[2] % self.size
            pool_mat = np.zeros((x.shape[0], x.shape[1], (x.shape[2] - sub) / self.size, (x.shape[2] - sub) / self.size))
        else:
            pool_mat = np.zeros((x.shape[0], x.shape[1], (x.shape[2]) / self.size, (x.shape[2]) / self.size))
        for i in range(np.shape(x)[0]):
            for j in range(np.shape(x)[1]):
                im_size = pool_mat.shape[2]
                for r in range(im_size):
                    for t in range(im_size):
                        matrix = x[i, j, self.size * r:self.size * r + self.size , self.size * t:self.size * t + self.size]
                        pool_num = np.max(matrix)
                        row_pos = self.size * r + np.where(matrix == pool_num)[0]
                        col_pos = self.size * t + np.where(matrix == pool_num)[1]
                        location[i, j, row_pos, col_pos] = 1
                        pool_mat[i, j, r, t] = pool_num
        self.locs = location
        return pool_mat


    def backward(self, y_grad):
        """
        Compute "backward" computation of maxpool layer

        Parameters
        ----------
        y_grad : np.array
            The gradient at the output

        Returns
        -------
        np.array
            The gradient at the input
        """
        grad_out = np.zeros_like(self.locs)
        for i in range(y_grad.shape[0]):
            for j in range(y_grad.shape[1]):
                for r in range(y_grad.shape[2]):
                    for t in range(y_grad.shape[3]):
                        matrix = self.locs[i, j, self.size * r:self.size * r + self.size, self.size * t:self.size * t + self.size]
                        grad = y_grad[i, j, r, t]
                        row_pos = self.size * r + np.where(matrix == 1)[0]
                        col_pos = self.size * t + np.where(matrix == 1)[1]
                        grad_out[i, j, row_pos, col_pos] = grad
        self.grad =grad_out
        return grad_out

    def update_param(self, lr):
        pass
