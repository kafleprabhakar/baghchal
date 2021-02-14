import os, sys
sys.path.append(os.path.join('..', 'backend'))

import torch as th
import random
import copy
import csv

from model import NeuralNet, PolicyModel
# from player import AutoPlayer
from pieces import Tiger, Goat, Empty
from utils import flatten_sa_pair, encode_state, get_move_idx, get_move_from_idx
from utils import Dataset


class BaseAgent():
    def __init__(self, board, piece, LR, train):
        self.board = board
        self.piece = piece
        self.LR = LR
        self.epsilon = 1
        self.eps_dec = 0.0001
        self.eps_min = 0.2
        self.experience = []
        self.data = None
        self.discount = 0.3
        self.train = train
        self.test_data = None

    def set_model(self, model):
        self.model = model
        self.optimizer = th.optim.Adam(self.model.parameters(), lr=self.LR)


    def get_all_moves(self):
        if self.piece == Goat and self.board.goats < 20:
            empty = self.board.get_positions(Empty)
            moves = [((i, j), '+') for i, j in empty]
        else:
            positions = self.board.get_positions(self.piece)
            moves = []
            for pos in positions:
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if self.board.check_move(pos, (dx, dy)):
                            moves.append((pos, (dx, dy)))
        
        return moves


    def make_move(self, move=None):
        if move is None:
            if random.random() < self.epsilon and self.train:
                move = self.get_random_move()
            else:
                move = self.get_best_move()

        prev_state = copy.deepcopy(self.board.values)

        r = self.board.make_move(*move)

        self.experience.append((prev_state, move, r))

    def get_random_move(self):
        valid_moves = self.get_all_moves()
        if len(valid_moves) == 0:
            raise Exception
        
        return random.choice(valid_moves)
    

    def get_best_move(self):
        valid_moves = self.get_moves_with_rewards()
        
        if len(valid_moves) == 0:
            raise Exception
        
        moves, rewards = zip(*valid_moves)
        rewards = th.Tensor(rewards)

        activation = th.nn.Softmax()
        scaled_rewards = activation(rewards)
        
        best_move_idx = th.multinomial(scaled_rewards, 1)
        
        return moves[best_move_idx]


    def save_experience(self, filename='experience.txt'):
        """
        Saves the experience of the current agent for the current round by
        appending it to the file specified as filename
        """
        with open(filename, 'a') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for exp in self.data:
                writer.writerow(exp)



class DQNAgent(BaseAgent):
    def __init__(self, board, piece, LR, train):
        super().__init__(board, piece, LR, train)
        self.model = NeuralNet()
        self.optimizer = th.optim.Adam(self.model.parameters(), lr=LR)
        self.loss_func = th.nn.MSELoss()


    def get_moves_with_rewards(self):
        valid_moves = self.get_all_moves()

        moves = []
        for move in valid_moves:
            inp = flatten_sa_pair((self.board.values, move)).float()
            reward = self.model(inp)
            moves.append((move, reward.item()))
        
        return moves

    # def get_best_move(self):
    #     moves = self.get_moves_with_rewards()

    #     if len(moves) == 0:
    #         raise Exception

    #     best_move = max(moves, key=lambda x: x[1])

    #     return best_move[0]


    def learn(self):
        self.optimizer.zero_grad()

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
            dataset = Dataset(test_file, featuresCols=range(84), targetCol=[84])
            self.test_data = th.utils.data.DataLoader(dataset, batch_size=20)
        
        total_loss = 0

        for data, target in self.test_data:
            data = data.float()
            pred = self.model(data)

            target = target.float().reshape(-1, 1)
            loss = self.loss_func(pred, target)

            total_loss += loss.item()

        return total_loss / len(dataset)


class TigerAgent(DQNAgent):
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


class GoatAgent(DQNAgent):
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
        






class PolicyAgent(BaseAgent):
    def __init__(self, board, piece, LR, train):
        super().__init__(board, piece, LR, train)
        self.model = PolicyModel()
        self.optimizer = th.optim.Adam(self.model.parameters(), lr=LR)
        self.loss_func = th.nn.CrossEntropyLoss()


    def get_moves_with_rewards(self):
        valid_moves = self.get_all_moves()

        valid_moves_idx = map(get_move_idx, valid_moves)

        best_move = None
        inp = encode_state(self.board.values).float()
        pred = self.model(inp)

        valid_moves_prob = [[valid_moves[i], pred[idx].item()] for i, idx in enumerate(valid_moves_idx)]

        return valid_moves_prob
        
        # valid_moves = self.get_all_moves()

        # moves = []
        # for move in valid_moves:
        #     inp = flatten_sa_pair((self.board.values, move)).float()
        #     reward = self.model(inp)
        #     moves.append((move, reward.item()))
        
        # return moves


    def learn(self):
        self.optimizer.zero_grad()

        data = th.tensor(self.data)
        
        n_features = data.shape[1] - 2
        state, move_idx, reward = th.split(data, [n_features, 1, 1], dim=1)
        target = move_idx.flatten().long()

        pred = self.model(state.float())

        loss = self.loss_func(pred, target).to(self.model.device)
        actual_loss = th.mean(loss * reward)
        actual_loss.backward()


        self.optimizer.step()

        # Decrease the epsilon to do more exploitation
        self.epsilon -= self.eps_dec
        self.epsilon = max(self.epsilon, self.eps_min)

        return actual_loss.item()/len(self.data)
    
    def test(self, test_file):
        if not self.test_data:
            dataset = Dataset(test_file, featuresCols=range(0, 50), targetCol=[50, 51])
            self.test_data = th.utils.data.DataLoader(dataset, batch_size=20)
        
        total_loss = 0

        for data, target in self.test_data:
            data = data.float()
            pred = self.model(data)

            target, reward = th.split(target, [1, 1], dim=1)
            target = target.flatten()

            loss = self.loss_func(pred, target)

            total_loss += th.mean(loss * reward).item()

        return total_loss / len(dataset)


class GoatPolicyAgent(PolicyAgent):
    def __init__(self, board, LR=0.1, train=True):
        super().__init__(board, piece=Goat, LR=LR, train=train)
        self.discount = 0.2

    def prepare_data(self):
        pass


class TigerPolicyAgent(PolicyAgent):
    def __init__(self, board, LR=0.1, train=True):
        super().__init__(board, piece=Tiger, LR=LR, train=train)
        self.discount = 0.4

    def prepare_data(self):
        self.data = []
        total = 0
        for exp in self.experience[-1::-1]:
            state, move, reward = exp

            vector = encode_state(state).flatten().tolist()

            move_idx = get_move_idx(move)
            vector.append(move_idx)

            total = reward + self.discount * total
            vector.append(total)

            self.data.append(vector)
        
        self.data.reverse()