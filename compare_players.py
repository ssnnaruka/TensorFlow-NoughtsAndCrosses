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

players = args.players


def compare(p1_name, p2_name):
    win_count = [0]*3

    c1 = globals()[p1_name]
    p1 = c1()

    c2 = globals()[p2_name]
    p2 = c2()

    for i in range(0, games):

        g = Game(p1,p2)

        gameover = False
        while not gameover:
            gameover = g.advance()

        w = g.get_winner()
        win_count[w] += 1

    return win_count


rows = []

headers = ['p2 \\ p1']
headers.extend(players)
rows.append(headers)

for y in players:
    row = [y]
    for x in players:
        win_count = compare(x,y)
        row.append("{1} ({0})".format(*win_count[:2]))

    rows.append(row)

def output_table(rows):
    max_y_label = max([len(r[0]) for r in rows])
    row_format = "{{:<{0}}} ".format(max_y_label+1)
    row_format += "".join(["{{:>{0}}}".format(len(h)+1) for h in rows[0][1:]])
    for row in rows:
        print(row_format.format(*row))


print("Number of player 1 wins (draws), against player 2:")
output_table(rows)
