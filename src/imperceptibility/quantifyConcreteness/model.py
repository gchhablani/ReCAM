# Implements a two layer Neural Network

import torch


class TwoLayerNN(torch.nn.Module):
    def __init__(self, D_in=300, H1=128, H2=64, D_out=1):

        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H1)
        self.linear2 = torch.nn.Linear(H1, H2)
        self.linear3 = torch.nn.Linear(H2, D_out)
        self.relu = torch.nn.ReLU()

    def forward(self, x):

        x1 = self.relu(self.linear1(x))
        x2 = self.relu(self.linear2(x1))
        x_output = self.linear3(x2)
        return x_output
