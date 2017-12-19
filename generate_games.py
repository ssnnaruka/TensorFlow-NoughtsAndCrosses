from game import *
from player import *
from trained_player import *
import argparse
from os import linesep

parser = argparse.ArgumentParser(description='Play two Players against each other a large number of times.')
parser.add_argument('-g', '--games', type=int, default=100, help="Number of games to run.")
parser.add_argument('players', nargs='+', help="Player class names", default='RandomPlayer')

args = parser.parse_args()

games = args.games
p1_name = args.players[0]
p2_name = args.players[1] if len(args.players) > 1 else args.players[0]


c1 = globals()[p1_name]
p1 = c1()

c2 = globals()[p2_name]
p2 = c2()


win_count = [0]*3


filename = 'data/Match-{0}-{1}-{2}.csv'.format(p1.name(), p2.name(), games)

f = open(filename, 'w')

for i in range(0, games):

    g = Game(p1,p2)

    gameover = False
    while not gameover:
        gameover = g.advance()

    w = g.get_winner()
    win_count[w] += 1

    # Output journal data
    j = g.get_journal()

    for r in j:
        outrow = list(r[0])
        outrow.extend([r[1], r[2]])

        f.write(",".join([str(a) for a in outrow]))
        f.write(linesep)

f.close()


print("Player 1 ({0}): Won {1}".format(p1.name(), win_count[1]))
print("Player 2 ({0}): Won {1}".format(p2.name(), win_count[2]))
print("Draws: {0}".format(win_count[0]))

