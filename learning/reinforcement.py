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

def run_simulation(verbose=False, LRs=[0.025], save_model=[], learn=[], plot=True, save_experience=[]):
    """
    Runs the simulation with the given parameters
    """
    EPOCHS = 20000
    n_plots = 200
    moves_per_game = 20
    
    for LR in LRs:
        avg_rewards_goat = []
        avg_rewards_tiger = []
        avg_loss_goat = []
        avg_loss_tiger = []

        # Choosing which model to use
        # tigerModel = NeuralNet()
        tigerModel = th.load('model-tiger-big.pt')
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

                        if verbose:
                            sys.stdout.write("\033[6F")
                            print(brd)
                            time.sleep(0.5)
                    except: # When the player doesn't have a move
                        over = True
                        break
                if over:
                    break
            
            # Prepare data for learning or saving
            goat_.prepare_data()
            tiger_.prepare_data()

            # Save the experience from this iteration
            if Goat in save_experience:
                goat_.save_experience('experience-goat-4.txt')
            
            if Tiger in save_experience:
                tiger_.save_experience('experience-tiger-4.txt')

            # Aggregate the total reward in this round
            goat_reward = 0
            for exp in goat_.data:
                goat_reward += exp[-1]

            tiger_reward = 0
            for exp in tiger_.data:
                tiger_reward += exp[-1]

            avg_rewards_goat.append(goat_reward/len(goat_.data))
            avg_rewards_tiger.append(tiger_reward/len(tiger_.data))


            # Learn from this experience
            if Goat in learn:
                loss_goat = goat_.learn()
                avg_loss_goat.append(loss_goat)

            if Tiger in learn:
                loss_tiger = tiger_.learn()
                avg_loss_tiger.append(loss_tiger)
        
        # Plot the rewards vs iteration graph
        if plot:
            iters = range(EPOCHS)
            n = int(EPOCHS/n_plots)
            plt.plot(iters[::n], avg_rewards_goat[::n], 'b-')
            plt.plot(iters[::n], avg_rewards_tiger[::n], 'r-')

    
    if plot:
        # plt.legend(['LR: %f'% LR for LR in LRs])
        plt.show()

    # Save the currently trained model
    if Goat in save_model:
        th.save(goatModel, 'model-goat-new.pt')
    
    if Tiger in save_model:
        th.save(tigerModel, 'model-tiger-new.pt')

    

if __name__ == '__main__':
    # LRs = [0.000000001, 0.000001, 0.0001, 0.001, 0.01, 0.1]
    run_simulation(verbose=False, LRs=[0.025], save_model=[Goat], learn=[Goat], plot=True, save_experience=[Goat, Tiger])