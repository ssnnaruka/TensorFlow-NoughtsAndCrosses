from game import *
from player import *

# Comment out following import if it breaks due to TensorFlow not being available
from trained_player import *

import argparse

parser = argparse.ArgumentParser(description='Play two Players against each other once.')
parser.add_argument('players', nargs='+', help="Player class names (e.g. RandomPlayer, HumanPlayer)", default='RandomPlayer')

args = parser.parse_args()

p1_name = args.players[0]
p2_name = args.players[1] if len(args.players) > 1 else args.players[0]


# Instantiate Player classes based on names supplied on command line
c1 = globals()[p1_name]
p1 = c1()

c2 = globals()[p2_name]
p2 = c2()

g = Game(p1,p2)

gameover = False
while not gameover:
    gameover = g.advance()
    g.output()

print("Winner: %d" % g.get_winner())
