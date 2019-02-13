from dataset import cifar100
import numpy as np
from full import FullLayer
from conv import ConvLayer
from maxpool import MaxPoolLayer
from flatten import FlattenLayer
from softmax import SoftMaxLayer
from cross_entropy import CrossEntropyLayer
from sequential import Sequential
from relu import ReluLayer
import matplotlib.pyplot as plt
from time import time

(x_train,y_train),(x_test,y_test) = cifar100(1212149859)
model = Sequential(layers= (ConvLayer(3,16,3),
                            ReluLayer(),
                            MaxPoolLayer(),
                            ConvLayer(16,32,3),
                            ReluLayer(),
                            MaxPoolLayer(),
                            FlattenLayer(),
                            FullLayer(2048,4),
                            SoftMaxLayer()),
                   loss= CrossEntropyLayer())
t0 = time()
epo = 15
loss = model.fit(x_train,y_train,epochs = epo, lr = 0.1, batch_size=128)
space = np.arange(0,epo)
pred = model.predict(x_test)
y_test = np.argmax(y_test,axis = 1)
acc = np.mean(pred == y_test)
plt.plot(space,loss,c = 'r')
print("done in %0.3fs." % (time() - t0))
plt.figure()
plt.plot(space,loss,label= 'Accuracy =' + str(acc) + ' with lr = 0.1')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.savefig('loss_plot_new.png')

