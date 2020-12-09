from board import Board
from player import AutoPlayer
from pieces import Goat, Tiger
from utils import jsonify
from secret import SECRET_KEY

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
    turn = request.args['turn']
    game = session['board']
    brd = Board(values= game['config'], goats= game['goats'])
    
    if turn == 'G':
        player = AutoPlayer(brd, Goat)
    else:
        player = AutoPlayer(brd, Tiger)
    
    player.make_move()
    new_brd = jsonify(brd)
    session['board'] = new_brd

    return json.dumps(new_brd)


        
if __name__ == '__main__':
    app.run(port=5000, debug= True)