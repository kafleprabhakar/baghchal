import os, sys
sys.path.append(os.path.join('..', 'backend'))

import torch as th
import random
import copy
import csv

from model import NeuralNet
from player import AutoPlayer
from pieces import Tiger, Goat
from utils import flatten_sa_pair
from dataset import Dataset


class BaseAgent(AutoPlayer):
    def __init__(self, board, piece, LR, train):
        super().__init__(board, piece)
        self.LR = LR
        self.model = NeuralNet()
        self.optimizer = th.optim.Adam(self.model.parameters(), lr=LR)
        self.loss_func = th.nn.MSELoss()
        self.epsilon = 1
        self.eps_dec = 0.0005
        self.eps_min = 0.2
        self.experience = []
        self.data = None
        self.discount = 0.3
        self.train = train
        self.test_data = None

    def set_model(self, model):
        self.model = model
        self.optimizer = th.optim.Adam(self.model.parameters(), lr=self.LR)

    def make_move(self, move=None):
        if move is None:
            if random.random() < self.epsilon and self.train:
                move = self.get_random_move()
            else:
                move = self.get_best_move()

        prev_state = copy.deepcopy(self.board.values)

        r = self.board.make_move(*move)

        self.experience.append((prev_state, move, r))

    def get_moves_with_rewards(self):
        valid_moves = self.get_all_moves()

        moves = []
        for move in valid_moves:
            inp = flatten_sa_pair((self.board.values, move)).float()
            reward = self.model(inp)
            moves.append((move, reward.item()))
        
        return moves

    def get_random_move(self):
        valid_moves = self.get_all_moves()
        if len(valid_moves) == 0:
            raise Exception
        return random.choice(valid_moves)

    def get_best_move(self):
        moves = self.get_moves_with_rewards()

        if len(moves) == 0:
            raise Exception

        best_move = max(moves, key=lambda x: x[1])

        return best_move[0]
            
    def save_experience(self, filename='experience.txt'):
        """
        Saves the experience of the current agent for the current round by
        appending it to the file specified as filename
        """
        with open(filename, 'a') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for exp in self.data:
                writer.writerow(exp)

    def learn(self):
        self.optimizer.zero_grad()

        total_loss = 0
        data = th.tensor(self.data).float()
        
        n_features = data.shape[1] - 1
        features, target = th.split(data, [n_features, 1], dim=1)

        pred = self.model(features)
        target = target.reshape(-1, 1)

        loss = self.loss_func(pred, target).to(self.model.device)
        loss.backward()

        total_loss = loss.item()

        self.optimizer.step()

        # Decrease the epsilon to do more exploitation
        self.epsilon -= self.eps_dec
        self.epsilon = max(self.epsilon, self.eps_min)

        return total_loss/len(self.data)
    
    def test(self, test_file):
        if not self.test_data:
            dataset = Dataset(test_file)
            self.test_data = th.utils.data.DataLoader(dataset, batch_size=20)
        
        total_loss = 0

        for data, target in self.test_data:
            data = data.float()
            pred = self.model(data)

            target = target.float().reshape(-1, 1)
            loss = self.loss_func(pred, target)

            total_loss += loss.item()

        return total_loss / len(dataset)


class TigerAgent(BaseAgent):
    def __init__(self, board, LR=0.1, train=True):
        super().__init__(board, piece=Tiger, LR=LR, train=train)
        self.discount = 0.3


    def prepare_data(self):
        self.data = []
        total = 0
        for exp in self.experience[-1::-1]:
            vector = flatten_sa_pair(exp[:-1]).tolist()
            total = exp[2] + self.discount * total
            vector.append(total)
            self.data.append(vector)
        
        self.data.reverse()


class GoatAgent(BaseAgent):
    def __init__(self, board, LR=0.1, train=True):
        super().__init__(board, piece=Goat, LR=LR, train=train)
        self.discount = 0.2

    def prepare_data(self):
        raw = []
        self.data = []
        prev = self.experience[-1][-1]

        for exp in self.experience[-1::-1]:
            reward = 10 - (exp[2] - prev) * 9
            raw.append((*exp[:-1], reward))
            prev = exp[2]
        
        total = 0
        for exp in raw[:-1]:
            vector = flatten_sa_pair(exp[:-1]).tolist()
            total = exp[-1] + self.discount * total
            vector.append(total)
            self.data.append(vector)

        self.data.reverse()
        