import os, sys
sys.path.append(os.path.join('..', 'backend'))
import time
import matplotlib.pyplot as plt
import torch as th
from tqdm import tqdm
import csv

from agent import TigerAgent, GoatAgent
from board import Board
from player import AutoPlayer
from pieces import Goat, Tiger
from model import NeuralNet
from utils import flatten_sa_pair

def run_simulation():
    # LRs = [0.01, 1, 5]
    LRs = [0.01]
    # LRs = [0.000000001, 0.000001, 0.0001, 0.001, 0.01, 0.1]
    all_experiences = []
    for LR in LRs:
        avg_rewards = []
        avg_loss = []
        tigerModel = NeuralNet()
        # tigerModel = th.load('tigerModel.pt')
        goatModel = NeuralNet()
        EPOCHS = 20000

        for i in tqdm(range(EPOCHS)):
            brd = Board()
            brd.init_game()

            alice = GoatAgent(brd, LR=LR)
            alice.set_model(goatModel)
            bob = TigerAgent(brd, LR)
            bob.set_model(tigerModel)

            players = [alice, bob]
            
            if i != 0:
                sys.stdout.write("\033[6F")

            print(brd)
            over = False
            for _ in range(40):
                for player in players:
                    try:
                        player.make_move()
                        sys.stdout.write("\033[6F")
                        print(brd)
                        # time.sleep(0.5)
                    except:
                        over = True
                        break
                if over:
                    break
            
            total_reward = 0

            bob.prepare_data()
            for exp in bob.data:
                total_reward += exp[2]
                vector = flatten_sa_pair(exp[:2]).tolist()
                all_experiences.append((vector, exp[2]))


            avg_rewards.append(total_reward/len(bob.experience))

            # alice.learn()
            # loss = bob.learn()
            # avg_loss.append(loss)
        
        # print('average rewards: ', avg_rewards)
        iters = range(EPOCHS)
        n = int(EPOCHS/50)
        plt.plot(iters[::n], avg_rewards[::n])
    plt.legend(['LR: %f'% LR for LR in LRs])
    plt.show()

    # th.save(tigerModel, 'tigerModel.pt')

    with open('experience.txt', 'w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for exp in all_experiences:
            writer.writerow(exp)

if __name__ == '__main__':
    run_simulation()