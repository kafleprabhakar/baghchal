import torch as th
from utils import flatten_sa_pair

class NeuralNet(th.nn.Module):
    def __init__(self, input_dims=275, output_dims=1):
        super(NeuralNet, self).__init__()
        self.layers = th.nn.Sequential(
            th.nn.Linear(input_dims, 64),
            th.nn.ReLU(),
            # th.nn.Linear(128, 32),
            # th.nn.ReLU(),
            th.nn.Linear(64, 1)
        )
        self.device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, data):
        inp = flatten_sa_pair(data).float().to(self.device)
        return self.layers(inp)