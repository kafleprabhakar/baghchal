import os, sys
sys.path.append(os.path.join('..', 'backend'))

from board import Board
# from player import AutoPlayer
from agent import GoatAgent, TigerAgent
from pieces import Goat, Tiger
from utils import jsonify
from secret import SECRET_KEY

import torch as th
from flask import Flask, session, request
from flask_cors import CORS, cross_origin
import json

app = Flask(__name__)
app.secret_key = SECRET_KEY
CORS(app, supports_credentials=True)
# app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/new_board', methods= ['GET'])
def new_board():
    brd = Board()
    brd.init_game()
    session['board'] = jsonify(brd)

    return json.dumps(jsonify(brd))

@app.route('/next_move', methods= ['GET'])
def next_move():
    brd, player = get_player_and_board()
    
    player.make_move()
    new_brd = jsonify(brd)
    session['board'] = new_brd

    return json.dumps(new_brd)

@app.route('/make_move', methods= ['GET'])
def make_move():
    brd, player = get_player_and_board()
    x = request.args.get('x', type=int)
    y = request.args.get('y', type=int)
    dx = request.args.get('dx', type=int)
    dy = request.args.get('dy', type=int)
    
    player.make_move(((x, y), (dx, dy)))

    new_brd = jsonify(brd)
    session['board'] = new_brd

    return json.dumps(new_brd)

@app.route('/get_best_moves', methods= ['GET'])
def get_best_moves():
    _, player = get_player_and_board()
    moves = player.get_moves_with_rewards()

    return json.dumps(moves)


def get_player_and_board():
    game = session['board']
    brd = Board(values= game['config'], goats= game['goats'])
    turn = request.args.get('turn', type=str)

    if turn == 'G':
        player = GoatAgent(brd, train=False)
        model = th.load('model-goat-new.pt')
    else:
        player = TigerAgent(brd, train=False)
        model = th.load('model-tiger-big.pt')
    
    player.set_model(model)

    return brd, player


        
if __name__ == '__main__':
    app.run(port=5000, debug= True)