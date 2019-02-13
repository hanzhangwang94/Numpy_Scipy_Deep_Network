import numpy as np
import scipy.signal


class ConvLayer(object):
    def __init__(self, n_i, n_o, h):
        """
        Convolutional layer

        Parameters
        ----------
        n_i : integer
            The number of input channels
        n_o : integer
            The number of output channels
        h : integer
            The size of the filter
        """
        # glorot initialization
        self.n_i = n_i
        self.n_o = n_o
        self.h = h
        self.W = np.sqrt(np.sqrt(2./(self.n_i + self.n_o + 2 * self.h)))*np.random.randn(self.n_o,self.n_i,self.h,self.h)
        self.b = np.zeros((1,self.n_o))
        self.W_grad = None
        self.b_grad = None

    def forward(self, x):
        """
        Compute "forward" computation of convolutional layer

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of input channels x number of rows x number of columns

        Returns
        -------
        np.array
            The output of the convolutiona

        Stores
        -------
        self.x : np.array
             The input data (need to store for backwards pass)
        """
        self.x = x
        pad_num = (np.shape(self.W)[2] - 1) / 2 # Here is the zero padding of our input, where pad_num = (filter_size - stride)/2
        h_o = (np.shape(x)[2] + 2*pad_num - np.shape(self.W)[2]) + 1 #This is the row and coloumn of output
        output = np.zeros((np.shape(x)[0],self.n_o,h_o,h_o)) # Define the shape of output
        for j in range(np.shape(x)[0]):
            for r in range(self.n_o):
                for i in range(self.n_i):
                    x_new = x[j,i,:,:]
                    x_pad = np.pad(x_new, ((pad_num, pad_num), (pad_num, pad_num)), 'constant', constant_values=0)
                    out = scipy.signal.correlate(x_pad,self.W[r,i,:,:],mode ='valid')
                    # print('bad',out)
                    output[j,r,:,:] +=out
                    # print('good',output)
        for i in range(self.n_o):
            output[:,i,:,:] += self.b[:,i]
        # print('final',output)
        return output


    def backward(self, y_grad):
        """
        Compute "backward" computation of convolutional layer

        Parameters
        ----------
        y_grad : np.array
            The gradient at the output

        Returns
        -------
        np.array
            The gradient at the input

        Stores
        -------
        self.b_grad : np.array
             The gradient with respect to b (same dimensions as self.b)
        self.w_grad : np.array
             The gradient with respect to W (same dimensions as self.W
        """
        out_b = np.sum(y_grad, axis = 2).reshape(np.shape(y_grad)[0],np.shape(y_grad)[1],1,np.shape(y_grad)[3])
        out_b = np.sum(out_b, axis = 3).reshape(np.shape(y_grad)[0], np.shape(y_grad)[1], 1, 1)
        self.b_grad = np.sum(out_b, axis = 0).reshape(1,np.shape(y_grad)[1])



        pad_num = (np.shape(self.W)[2] - 1) / 2
        x_pad = np.pad(self.x, ((0, 0), (0, 0), (pad_num, pad_num), (pad_num, pad_num)), mode='constant', constant_values=0)
        self.W_grad = np.zeros_like(self.W)
        for j in range(np.shape(self.x)[1]):
            for r in range(np.shape(y_grad)[1]):
                out_w = scipy.signal.correlate(x_pad[:, j, :, :], y_grad[:, r, :, :], mode='valid').reshape(
                    np.shape(self.W)[2], np.shape(self.W)[2])
                self.W_grad[r, j, :, :] = out_w



        outcome = np.zeros_like(self.x)
        for i in range(np.shape(y_grad)[0]):
            channel = np.zeros((np.shape(self.W)[0], np.shape(self.W)[1], np.shape(self.x)[2], np.shape(self.x)[2]))
            for j in range(np.shape(self.W)[0]):
                for r in range(np.shape(self.W)[1]):
                    out = scipy.signal.convolve(y_grad[i, j, :, :], self.W[j, r, :, :], mode='same')
                    channel[j, r, :, :] = out

            channel = np.sum(channel, axis=0).reshape(np.shape(self.W)[1], np.shape(self.x)[2], np.shape(self.x)[2])
            outcome[i, :, :, :] = channel
        x_grad = outcome
        self.x_grad = outcome
        return x_grad


    def update_param(self, lr):
        """
        Update the parameters with learning rate lr

        Parameters
        ----------
        lr : floating point
            The learning rate

        Stores
        -------
        self.W : np.array
             The updated value for self.W
        self.b : np.array
             The updated value for self.b
        """
        self.W = self.W - lr * self.W_grad
        self.b = self.b - lr * self.b_grad
