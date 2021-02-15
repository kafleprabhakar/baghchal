import torch as th
from utils import flatten_sa_pair

class NeuralNet(th.nn.Module):
    def __init__(self, input_dims=84, output_dims=1):
        super(NeuralNet, self).__init__()
        self.layers = th.nn.Sequential(
            th.nn.Linear(input_dims, 48),
            th.nn.ReLU(),
            th.nn.Linear(48, 16),
            th.nn.ReLU(),
            th.nn.Linear(16, 1),
        )
        self.boardNet = th.nn.Sequential(
            th.nn.Conv2d(3, 5, 3, 1),
            th.nn.ReLU(),
            th.nn.BatchNorm2d(5),
            th.nn.Flatten()
        )
        self.actionNet = th.nn.Sequential(
            th.nn.Linear(9, 16),
            th.nn.ReLU()
        )
        self.final = th.nn.Sequential(
            th.nn.Linear(61, 16),
            th.nn.ReLU(),
            th.nn.Linear(16, output_dims)
        )
        self.device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, data):
        if len(data.shape) == 1:
            data = data.reshape(1, -1)

        brd, move = th.split(data, [75, 9], dim=1)
        brd = brd.reshape(-1, 3, 5, 5)

        brd.to(self.device)
        move.to(self.device)

        x_brd = self.boardNet(brd)
        x_act = self.actionNet(move)

        x = th.cat((x_brd, x_act), 1)

        return self.final(x)

class PolicyModel(th.nn.Module):
    def __init__(self, output_dims=25*9):
        super(PolicyModel, self).__init__()
        self.layers = th.nn.Sequential(
            th.nn.Conv2d(2, 8, 3, 1),
            th.nn.ReLU(),
            th.nn.BatchNorm2d(8),
            th.nn.Flatten(),
            th.nn.Linear(72, 64),
            th.nn.ReLU(),
            # th.nn.Linear(128, 64),
            # th.nn.ReLU(),
            th.nn.Linear(64, output_dims)
        )

        self.device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, data):
        data.to(self.device)
        if len(data.shape) < 3:
            # print('reshaping data')
            data = data.reshape(-1, 2, 5, 5)
        return self.layers(data)