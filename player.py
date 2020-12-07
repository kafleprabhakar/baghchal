from pieces import Goat, Tiger
import random

class Player:
    def __init__(self, board, piece):
        self.board = board
        self.piece = piece

class AutoPlayer(Player):
    def make_move(self):
        if self.piece == Goat and self.board.goats < 20:
            self.add_goat()
        else:
            positions = self.board.get_positions(self.piece)
            success = False
            while not success:
                try:
                    pos = random.choice(positions)
                    # pos = (0,0)
                    dx = random.randint(-1, 1)
                    dy = random.randint(-1, 1)
                    # dx, dy = 1, 1
                    self.board.make_move(pos, (dx, dy))
                    success = True
                except Exception as e:
                    # print('this exception', e)
                    pass

    def add_goat(self):
        # print('adding goat')
        success = False
        while not success:
            try:
                x = random.randint(0, 4)
                y = random.randint(0, 4)
                self.board.add_goat((x, y))
                success = True
            except Exception as e:
                # print('while adding goat', e)
                pass

