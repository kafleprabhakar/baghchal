from pieces import Goat, Tiger, Empty, Piece
from config import GOAT_REWARDS, TIGER_REWARDS

class Board:
    def __init__(self, size = 5, values= None, goats= 0):
        conf = []
        if values:
            for i, row in enumerate(values):
                r = []
                for j, val in enumerate(row):
                    if val == 'G':
                        v = Goat
                    elif val == 'T':
                        v = Tiger
                    else:
                        v = Empty
                    r.append(v((j, i)))
                conf.append(r)
        
        self.goats = goats
        self.size = len(conf) if values else size
        self.values = conf if values else [[Empty((i, j)) for i in range(size)] for j in range(size)]

    def init_game(self):
        last = self.size - 1
        for loc in [(0, 0), (last, 0), (0, last), (last, last)]:
            self.values[loc[0]][loc[1]] = Tiger(loc)

    def __str__(self):
        res = ''
        for row in self.values:
            for val in row:
                res += str(val) + ' '
            res += '\n'
        return res

    def add_goat(self, position):
        pos = self.check_location(position)

        if not isinstance(pos, Empty):
            raise Exception('Position already occupied')

        if self.goats >= 20:
            raise Exception('Too many goats. Cannot place one more')
        
        self.goats += 1
        x, y = position
        self.values[y][x] = Goat((x, y))
    
    def get_positions(self, piece):
        locations = []
        for i, row in enumerate(self.values):
            for j, col in enumerate(row):
                if isinstance(col, piece):
                    locations.append((j, i))
        return locations

    def check_location(self, location):
        assert isinstance(location, tuple) and len(location) == 2

        for c in location:
            assert 0 <= c < self.size
        
        x, y = location
        return self.values[y][x]

    def check_move(self, location, direction):
        try:
            self.make_move(location, direction, check=True)
            return True
        except Exception as e:
            return False

    def make_move(self, location, direction, check=False):
        """
        Given a location and a direction from that location, moves the
        piece at that location in the direction if it is a valid move.

        check = True doesn't make the move, but only checks if the move is valid
        """
        assert isinstance(location, tuple) and len(location) == 2
        assert (isinstance(direction, tuple) and len(direction) == 2 and all(-1 <= dir_ <= 1 for dir_ in direction)) or direction == '+'
        assert direction != (0,0)
        
        piece = self.check_location(location)

        x, y = location
        if isinstance(piece, Empty):
            if direction == '+':
                self.add_goat(location)
                return GOAT_REWARDS['ADD']
            else:
                raise Exception('Location empty')

        # If it is in a position where it can't make diagonal moves
        if sum(location) % 2 == 1 and sum(abs(i) for i in direction) == 2:
            raise Exception('Invalid move')

        to = tuple(location[i] + direction[i] for i in range(2))
        # print('to', to)
        target = self.check_location(to)

        # print('the target piece', target)
        if isinstance(target, Tiger):
            raise Exception('Invalid Move')

        if isinstance(target, Goat):
            landing = tuple(location[i] + direction[i] * 2 for i in range(2))
            land = self.check_location(landing)
            if not isinstance(land, Empty) or isinstance(piece, Goat):
                raise Exception('Invalid Move')
            
            if not check:
                self.values[y][x] = Empty((x, y))
                self.values[to[1]][to[0]] = Empty(to)
                self.values[landing[1]][landing[0]] = piece
                piece.move(landing)
                return TIGER_REWARDS['CAPTURE']
        elif not check:
            self.values[y][x] = Empty((x, y))
            self.values[to[1]][to[0]] = piece
            piece.move(to)

            return GOAT_REWARDS['MOVE']