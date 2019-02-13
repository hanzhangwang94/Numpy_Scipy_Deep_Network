from dataset import cifar100
import numpy as np
from full import FullLayer
from softmax import SoftMaxLayer
from cross_entropy import CrossEntropyLayer
from sequential import Sequential
from relu import ReluLayer
import matplotlib.pyplot as plt
(x_train,y_train),(x_test,y_test) = cifar100(1212149859)

model = Sequential(layers = ( FullLayer(32*32*3,256),
                                ReluLayer(),
                                FullLayer(256,4),
                             SoftMaxLayer()),
                   loss = CrossEntropyLayer())


model.fit(x_train,y_train,epochs =15, lr = 0.48, batch_size=128)
pred = model.predict(x_test)
acc = np.mean(pred == y_test)

print('Accuracy = %f'%acc)
index_0 = np.where(y_test == 0)[0]
index_1 = np.where(y_test == 1)[0]
index_2 = np.where(y_test == 2)[0]
index_3 = np.where(y_test == 3)[0]
acc0 = np.mean(y_test[index_0] == pred[index_0])
acc1 = np.mean(y_test[index_1] == pred[index_1])
acc2 = np.mean(y_test[index_2] == pred[index_2])
acc3 = np.mean(y_test[index_3] == pred[index_3])
print('class0 accuracy =%f'%acc0)
print('class1 accuracy =%f'%acc1)
print('class2 accuracy =%f'%acc2)
print('class3 accuracy =%f'%acc3)