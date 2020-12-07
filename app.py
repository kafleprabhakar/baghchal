from board import Board
from player import AutoPlayer
import random
import sys
import time
from pieces import Goat, Tiger

brd = Board()
brd.init_game()
alice = AutoPlayer(brd, Goat)
bob = AutoPlayer(brd, Tiger)
players = [alice, bob]

print(brd)

for _ in range(30):
    for player in players:
        player.make_move()
        sys.stdout.write("\033[6F")
        print(brd)
        time.sleep(.5)
        
    # x = random.randint(0, 5)
    # y = random.randint(0, 5)
    # dx = random.randint(-1, 1)
    # dy = random.randint(-1, 1)
    # # if x == 0 and y == 0:
    # #     print('location', (dx, dy))
    # try:
    #     brd.make_move((x, y), (dx, dy))
    #     sys.stdout.write("\033[6F")
    #     print(brd)
    #     time.sleep(.5)
    # except Exception as e:
    #     # print(e)
    #     pass
        
