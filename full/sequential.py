from __future__ import print_function
import numpy as np
from sklearn.utils import shuffle

class Sequential(object):
    def __init__(self, layers, loss):
        """
        Sequential model

        Implements a sequence of layers

        Parameters
        ----------
        layers : list of layer objects
        loss : loss object
        """
        self.layers = layers
        self.loss = loss

    def forward(self, x, target=None):
        """
        Forward pass through all layers
        
        if target is not none, then also do loss layer

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number of features
        target : np.array
            The target data of size number of training samples x number of features (one-hot)

        Returns
        -------
        np.array
            The output of the model
        """
        for i in self.layers:
            y = i.forward(x)
            x = y

        if target is None:
            out = y
        else:
            out = self.loss.forward(y,target)
        return out



    def backward(self):
        """
        Compute "backward" computation of fully connected layer

        Returns
        -------
        np.array
            The gradient at the input

        """
        grad = self.loss.backward()
        for i in reversed(self.layers):
            grad_out = i.backward(grad)
            grad = grad_out
        return grad

    def update_param(self, lr):
        """
        Update the parameters with learning rate lr

        Parameters
        ----------
        lr : floating point
            The learning rate
        """
        for i in self.layers:
            i.update_param(lr)


    def fit(self, x, y, epochs=10, lr=0.1, batch_size=128):
        """
        Fit parameters of all layers using batches

        Parameters
        ----------
        x : numpy matrix
            Training data (number of samples x number of features)
        y : numpy matrix
            Training labels (number of samples x number of features) (one-hot)
        epochs: integer
            Number of epochs to run (1 epoch = 1 pass through entire data)
        lr: float
            Learning rate
        batch_size: integer
            Number of data samples per batch of gradient descent
        """
        # for i in range(epochs):
        #     x_batch = x[i + i * (batch_size - 1):(batch_size) + i * batch_size, :]
        #     y_batch = y[i + i * (batch_size - 1):(batch_size) + i * batch_size, :]
        #     self.forward(x_batch, y_batch)
        #     self.backward()
        #     self.update_param(lr)
        loss = []
        for i in range(epochs):
            train = np.hstack((x,y))
            train_new = shuffle(train,random_state = epochs)
            x_train = train_new[:,0:-4]
            y_train = train_new[:, -4:]
            loop = int(np.shape(x)[0]/batch_size)
            for j in range(loop):
                x_batch = x_train[j + j * (batch_size - 1):(batch_size) + j * batch_size, :]
                y_batch = y_train[j + j * (batch_size - 1):(batch_size) + j * batch_size, :]
                out = self.forward(x_batch,y_batch)
                self.backward()
                self.update_param(lr)
            loss.append(out)
        loss = np.array(loss)
        return loss



    def predict(self, x):
        """
        Return class prediction with input x

        Parameters
        ----------
        x : numpy matrix
            Testing data data (number of samples x number of features)

        Returns
        -------
        np.array
            The output of the model (integer class predictions)
        """
        output = self.forward(x)
        prediction = np.argmax(output,axis = 1)
        return prediction
