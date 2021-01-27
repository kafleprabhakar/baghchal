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
        self.device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, data):
        data.to(self.device)
        return self.layers(data)