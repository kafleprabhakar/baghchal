def jsonify(board):
    config = [[str(val) for val in row] for row in board.values]
    return {'config': config, 'goats': board.goats}