import os, sys
sys.path.append(os.path.join('..', 'backend'))
import time
import matplotlib.pyplot as plt
import torch as th
from tqdm import tqdm
import csv

from agent import TigerAgent, GoatAgent, GoatPolicyAgent, TigerPolicyAgent
from board import Board
from pieces import Goat, Tiger
from model import NeuralNet, PolicyModel
from utils import flatten_sa_pair, PeriodicPlotter

def run_simulation(verbose=False, LRs=[0.025], save_model=[], learn=[], plot=True, save_experience=[]):
    """
    Runs the simulation with the given parameters
    """
    EPOCHS = 100
    n_plots = 4
    n = int(EPOCHS/n_plots)
    
    moves_per_game = 20
    
    for LR in LRs:
        avg_rewards_goat = []
        avg_rewards_tiger = []
        avg_loss_goat = []
        avg_loss_tiger = []

        if plot and len(learn) != 0:
            plt.ion()
            iters = []
            tiger_plot = PeriodicPlotter('r', x_lim=(0, EPOCHS), y_lim=(0, 10))
            goat_plot = PeriodicPlotter('b', x_lim=(0, EPOCHS), y_lim=(0, 10))


        # Choosing which model to use
        tigerModel = NeuralNet()
        # tigerModel = th.load('model-tiger-big.pt')
        goatModel = NeuralNet()
        # goatModel = th.load('goatModel-learn.pt')

        for i in tqdm(range(EPOCHS)):
            # Initialize this round of game
            brd = Board()
            brd.init_game()

            # Initialize the agents and set their model
            goat_ = GoatAgent(brd, LR=LR)
            goat_.set_model(goatModel)

            tiger_ = TigerAgent(brd, LR)
            tiger_.set_model(tigerModel)

            players = [goat_, tiger_]
            
            if verbose:
                if i != 0:
                    sys.stdout.write("\033[6F")

                print(brd)
            
            over = False # Whether the current game is over or not
            for _ in range(moves_per_game):

                for player in players:
                    try:
                        player.make_move()
                        # print('player', player)

                        if verbose:
                            sys.stdout.write("\033[6F")
                            print(brd)
                            time.sleep(0.5)
                    except Exception as e: # When the player doesn't have a move
                        over = True
                        break
                if over:
                    break


            # Save the experience from this iteration
            if Goat in save_experience:
                goat_.save_experience('experience-goat-dqn-test.txt')
            
            if Tiger in save_experience:
                tiger_.save_experience('experience-tiger-dqn-test.txt')

            # Aggregate the total reward in this round


            # avg_rewards_goat.append(goat_reward/len(goat_.data))


            # Learn from this experience
            if plot and i % n == 0:
                iters.append(i)

            if Goat in learn:
                goat_.prepare_data()
                goat_.learn()

                goat_reward = 0
                for exp in goat_.data:
                    goat_reward += exp[-1]
                
                if plot and i % n == 0:
                    loss_goat = goat_.test('experience-goat-dqn-test.txt')
                    avg_loss_goat.append(loss_goat)
                    goat_plot.plot(iters, avg_loss_goat)

            if Tiger in learn:
                tiger_.prepare_data()
                tiger_.learn()
                
                tiger_reward = 0
                for exp in tiger_.data:
                    tiger_reward += exp[-1]

                if plot and i % n == 0:
                    loss_tiger = tiger_.test('experience-tiger-test.txt')
                    avg_rewards_tiger.append(tiger_reward/len(tiger_.data))
                    avg_loss_tiger.append(loss_tiger)
                    tiger_plot.plot(iters, avg_loss_tiger)
        

    
    if plot:
        # plt.legend(['LR: %f'% LR for LR in LRs])
        plt.show()
        plt.savefig('result1.png')

    # Save the currently trained model
    if Goat in save_model:
        th.save(goatModel, 'model-goat-dqn.pt')
    
    if Tiger in save_model:
        th.save(tigerModel, 'model-tiger-dqn.pt')

    

if __name__ == '__main__':
    run_simulation(verbose=False, LRs=[0.0025], save_model=[], learn=[Tiger], plot=True, save_experience=[])