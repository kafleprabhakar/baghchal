import os, sys
sys.path.append(os.path.join('..', 'backend'))

import torch as th
import random
import copy

from model import NeuralNet
from player import AutoPlayer
from pieces import Tiger, Goat


class BaseAgent(AutoPlayer):
    def __init__(self, board, piece, LR, train):
        super().__init__(board, piece)
        self.LR = LR
        self.model = NeuralNet()
        self.optimizer = th.optim.Adam(self.model.parameters(), lr=LR)
        self.loss_func = th.nn.MSELoss()
        self.epsilon = 1
        self.eps_dec = 0.002
        self.eps_min = 0.2
        self.experience = []
        self.data = None
        self.discount = 0.3
        self.train = train

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
            reward = self.model((self.board.values, move))
            moves.append((move, reward.item()))
        
        return moves

    def get_random_move(self):
        valid_moves = self.get_all_moves()
        if len(valid_moves) == 0:
            raise Exception
        return random.choice(valid_moves)

    def get_best_move(self):
        valid_moves = self.get_all_moves()
        if len(valid_moves) == 0:
            raise Exception
        best_move = None
        best_reward = 0
        for move in valid_moves:
            reward = self.model((self.board.values, move))
            if best_move is None or reward > best_reward:
                best_move = move
                best_reward = reward
        
        return best_move


    def prepare_data(self):
        self.data = []
        total = 0
        for exp in self.experience[-1::-1]:
            total = exp[2] + self.discount * total
            self.data.append((*exp[:-1], total))
        
        self.data.reverse()
            

    def learn(self):
        total_loss = 0
        for *inp, target in self.experience:
            self.optimizer.zero_grad()


            inp = flatten_sa_pair(inp).float()
            pred = self.model(inp)

            target = th.tensor(target).float().reshape(1, 1)
            loss = self.loss_func(pred, target).to(self.model.device)
            loss.backward()

            total_loss += loss.item()

            self.optimizer.step()
        
        self.epsilon -= self.eps_dec
        self.epsilon = max(self.epsilon, self.eps_min)

        return total_loss/len(self.experience)


class TigerAgent(BaseAgent):
    def __init__(self, board, LR=0.1, train=True):
        super().__init__(board, piece=Tiger, LR=LR, train=train)


class GoatAgent(BaseAgent):
    def __init__(self, board, LR=0.1, train=True):
        super().__init__(board, piece=Goat, LR=LR, train=train)