import os, sys
sys.path.append(os.path.join('..', 'backend'))

import matplotlib.pyplot as plt
import torch as th
import pandas as pd
from pieces import Goat, Tiger, Empty

moves = [(i, j) for i in range(-1, 2) for j in range(-1, 2)]
moves[4] = '+' # Corresponding to move (0, 0)


def encode_state(state):
    """
    Given a state of the board, returns a one hot encoding of the board configuration
    """
    state_vector = []
    for row in state:
        vector_row = []
        for val in row:
            if isinstance(val, Goat):
                one_hot = [1, 0]
            elif isinstance(val, Tiger):
                one_hot = [0, 1]
            else:
                one_hot = [0, 0]
            vector_row.append(one_hot)
        state_vector.append(vector_row)

    # Transpose to a 2 * 5 * 5 tensor representing the position of
    # tiger and goats in the 5 * 5 board
    return th.tensor(state_vector).transpose(0, 2).transpose(1, 2)


def flatten_sa_pair(inp):
    """
    Given a state, action pair at a form of tuples, flattens and
    one hot encodes it
    """
    state, action = inp
    state_vector = encode_state(state).flatten().reshape(1, -1)
    
    # move_vector = [1] if move == '+' else [0] # For Goat model

    pos, move = action
    pos_vector = [0 for _ in range(25)]
    # action_vector = [[0 for _ in range(9)] for _ in range(25)]
    pos_idx = pos[1] * 5 + pos[0]
    pos_vector[pos_idx] = 1

    move_vector = [0 for _ in range(9)]
    move_idx = moves.index(move)
    move_vector[move_idx] = 1
    # action_vector[pos_idx][action_idx] = 1

    pos_vector = th.tensor(pos_vector).reshape(1, -1)
    move_vector = th.tensor(move_vector).reshape(1, -1)

    return th.cat((state_vector, pos_vector, move_vector), 1).reshape(-1)

def flatten_sa_pair_old(inp):
    """
    Given a state, action pair at a form of tuples, flattens and
    one hot encodes it
    """
    state, action = inp
    state_vector = []
    for row in state:
        for val in row:
            if isinstance(val, Goat):
                one_hot = [1, 0]
            elif isinstance(val, Tiger):
                one_hot = [0, 1]
            else:
                one_hot = [0, 0]
            state_vector.append(one_hot)
    
    # move_vector = [1] if move == '+' else [0] # For Goat model

    pos, move = action
    action_vector = [[0 for _ in range(9)] for _ in range(25)]
    pos_idx = pos[1] * 5 + pos[0]
    action_idx = moves.index(move)
    action_vector[pos_idx][action_idx] = 1

    state_vector = th.tensor(state_vector).reshape(1, -1)
    action_vector = th.tensor(action_vector).reshape(1, -1)

    return th.cat((state_vector, action_vector), 1).reshape(-1)


def get_move_idx(move):
    pos, direction = move

    x, y = pos
    pos_idx = y * 5 + x

    dx, dy = (0,0) if direction == '+' else direction
    dir_idx = dy * 3 + dx

    return pos_idx * 9 + dir_idx

def get_move_from_idx(idx):
    pos_idx = idx // 9
    dir_idx = idx % 9

    pos = (pos_idx // 25, pos_idx % 25)
    direction = (dir_idx // 3, dir_idx % 3)

    return (pos, direction)

def jsonify(board):
    config = [[str(val) for val in row] for row in board.values]
    return {'config': config, 'goats': board.goats}


class PeriodicPlotter():
    def __init__(self, color='r', x_lim=None, y_lim=None):
        plt.ion()
        self.line, = plt.plot([], [], color + '-')
        self.line.axes.set_xlim(*x_lim)
        self.line.axes.set_ylim(*y_lim)

    def plot(self, x, y):
        self.line.set_xdata(x)
        self.line.set_ydata(y)
        plt.draw()
        plt.pause(0.01)



class Dataset(th.utils.data.Dataset):
    def __init__(self, filename, featuresCols, targetCol):
        dataset = pd.read_csv(filename)
        
        self.targets = dataset.iloc[:, targetCol]
        self.features = dataset.iloc[:, featuresCols]
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        """
        Get the item at specified index
        """
        if th.is_tensor(idx):
            idx = idx.tolist()
        
        return th.from_numpy(self.features.iloc[idx].to_numpy()).type(th.FloatTensor),\
             th.tensor(self.targets.iloc[idx]).type(th.LongTensor)