from sklearn.model_selection import train_test_split
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class MLP(nn.Module):
    # Define multilayer Perceptrons
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(3, 3)
        self.fc2 = nn.Linear(3, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = self.fc2(X)
        X = self.softmax(X)
        return X


r''' Data pre-processing '''
# Load dataset
data_feature, data_label = np.load('data/binary_mnist_data.npy'), np.load('data/binary_mnist_label.npy')
# Split the dataset into two groups
train_X, test_X, train_y, test_y = train_test_split(data_feature,
                                                    data_label, test_size=0.78)
# Transform data to the Pytorch's form
train_X = Variable(torch.Tensor(train_X).float())
test_X = Variable(torch.Tensor(test_X).float())
train_y = Variable(torch.Tensor(train_y).long())
test_y = Variable(torch.Tensor(test_y).long())

def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels]

# Transform labels into the one-hot form
train_y_onehot = one_hot_embedding(train_y, 2)

r''' Start training !'''
loss_record, train_record, test_record = [], [], []

net = MLP()
criterion = nn.MSELoss()  # mean square loss
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

for epoch in range(400):
    optimizer.zero_grad()
    out = net(train_X)
    loss = criterion(out, train_y_onehot)
    loss.backward()
    optimizer.step()

    if epoch % 1 == 0:
        print('epoch is %d || loss is %f'%(epoch, loss.data))
        loss_record.append(loss.data.numpy())
        with torch.no_grad():
            predict_test = net(test_X)
            _, predict_test_y = torch.max(predict_test, 1)
            correct_test = (predict_test_y == test_y).sum().item()
            predict_train = net(train_X)
            _, predict_train_y = torch.max(predict_train, 1)
            correct_train = (predict_train_y == train_y).sum().item()
        print('Train acc %f || Test acc %f' %(correct_test / len(test_y), correct_train / len(train_y)))
        train_record.append(correct_train / len(train_y))
        test_record.append(correct_test / len(test_y))

file_trainloss = 'trainloss_DNN'
file_testacc = 'testacc_DNN'
file_trainacc = 'trainacc_DNN'
np.save(file_trainacc, train_record)
np.save(file_testacc, test_record)
np.save(file_trainloss, loss_record)

os._exit(1)
