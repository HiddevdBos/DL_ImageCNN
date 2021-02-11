import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.nn import Linear, ReLU, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout


class CNN(Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )
        self.linear_layers = Sequential(Linear(4 * 7 * 7, 10))

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

def train_model(images, labels, epochs=20, learningRate=0.007):
    pass
    # x, y = matrices_to_tensors(x, y)
    # model = Net()
    # model = model.float()
    # optimizer = Adam(model.parameters(), lr=learningRate, weight_decay=l2_weight_decay)
    # criterion = CrossEntropyLoss()
    # loss_list = []
    # # batch sizes are calculated and sorted
    # if not batch_size or batch_size > x.shape[0]:
    #     batch_size = x.shape[0]
    # batch_num = x.shape[0] / batch_size
    # x = x.reshape(-1, batch_size, 1, 16, 15)
    # y = y.reshape(-1, batch_size)
    #
    # for epoch in range(0, epochs):
    #     # loop over the number of batches feeds in batch_size many images and performs backprop
    #     # then again and so on
    #     for i in range(0, int(batch_num)):
    #         # here we feed in training data and perform backprop according to the loss
    #         # run the forward pass
    #         outputs = model.forward(x[i])
    #         y = y.long()
    #         loss = criterion(outputs, y[i])
    #         loss_list.append(loss.item())
    #
    #         # backprop and perform Adam optimisation
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    # return model, loss_list

def test_model():
    pass
