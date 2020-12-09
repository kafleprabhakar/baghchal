class Piece:
    def __init__(self, location):
        self.location = location
    
    def move(self, location):
        self.location = location

class Goat(Piece):
    
    def __str__(self):
        return 'G'

class Tiger(Piece):
    def __str__(self):
        return 'T'
