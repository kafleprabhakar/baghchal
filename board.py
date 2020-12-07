from pieces import Goat, Tiger, Piece

class Board:
    def __init__(self, size = 5):
        self.goats = 0
        self.size = size
        self.values = [['_' for _ in range(size)] for _ in range(size)]

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

        if isinstance(pos, Piece):
            raise Exception('Position already occupied')

        if self.goats >= 20:
            raise Exception('Too many goats. Cannot place one more')
        
        self.goats += 1
        x, y = position
        self.values[x][y] = Goat((x, y))
    
    def get_positions(self, piece):
        locations = []
        for i, row in enumerate(self.values):
            for j, col in enumerate(row):
                if isinstance(col, piece):
                    locations.append((i, j))
        return locations

    def check_location(self, location):
        assert isinstance(location, tuple) and len(location) == 2

        for c in location:
            assert 0 <= c < self.size
        
        x, y = location
        return self.values[x][y]

    def make_move(self, location, direction):
        # print('the location', location)
        # print('the direction', direction)
        # assert isinstance(location, tuple) and len(location) == 2
        assert isinstance(direction, tuple) and len(direction) == 2
        assert direction != (0,0)
        piece = self.check_location(location)
        # print('the piece', piece)
        x, y = location
        if not isinstance(piece, Piece):
            raise Exception('Location empty')

        # print('the piece', piece)
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
            if isinstance(land, Piece) or isinstance(piece, Goat):
                raise Exception('Invalid Move')

            self.values[x][y] = '_'
            self.values[to[0]][to[1]] = '_'
            self.values[landing[0]][landing[1]] = piece
            piece.move(landing)
        else:
            self.values[x][y] = '_'
            self.values[to[0]][to[1]] = piece
            piece.move(to)
            # print('finished moving')