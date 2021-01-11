from pieces import Goat, Tiger, Empty
import random

class Player:
    def __init__(self, board, piece):
        self.board = board
        self.piece = piece
    
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

class AutoPlayer(Player):
    def make_random_move(self):
        if self.piece == Goat and self.board.goats < 20:
            self.add_goat()
            # return {'type': 'A', 'pos': pos}
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
                    # return (pos, (dx, dy))
                    # return {'type': 'M', 'pos': pos, 'direction': (dx, dy)}
                    success = True
                except Exception as e:
                    # print('this exception', e)
                    pass

    def add_goat(self):
        success = False
        while not success:
            try:
                x = random.randint(0, 4)
                y = random.randint(0, 4)
                self.board.add_goat((x, y))
                success = True
                # return (x, y)
            except Exception as e:
                # print('while adding goat', e)
                pass

    def make_move(self):
        return self.make_random_move()