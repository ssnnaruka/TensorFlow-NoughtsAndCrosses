import csv
import numpy as np

from trained_player import *

import argparse

parser = argparse.ArgumentParser(description='Train a TrainedPlayer and output a model class.')
parser.add_argument('-i', '--inputfile', type=argparse.FileType('r'),
                    default='data/Match-RandomPlayer-RandomPlayer-10000.csv',
                    help="CSV game data used to train.")
parser.add_argument('-b', '--batchsize', type=int, default=100,
                    help="Batch size for training.")
parser.add_argument('-e', '--epochs', type=int, default=1,
                    help="Number of epochs for training.")
parser.add_argument('players', nargs=1, help="TrainedPlayer class name", default='TrainedPlayer')

args = parser.parse_args()

p1_name = args.players[0]

c1 = globals()[p1_name]
p1 = c1()


inputfile = args.inputfile

batch_size = args.batchsize

epochs = args.epochs


data = []

csvreader = csv.reader(inputfile)
for row in csvreader:
    data.append([int(c) for c in row])


# Only keep rows where 'this' player ended up winning
data = [d for d in data if d[10] == 1]

def onehot(m):
    """
    Convert a move (0-8) into one-hot representation on the board
    E.g. onehot(3) => [0,0,0,1,0,0,0,0,0]
    """
    r = [0]*9
    r[m] = 1
    return r


all_xs = np.array([row[0:9] for row in data])

# One-hot
all_ys = np.array([onehot(row[9]) for row in data])


def train_model(steps, batch_size, epochs):
    global all_xs, all_ys

    model = p1

    with model.graph.as_default():

        init = tf.global_variables_initializer()

        sess = tf.Session()

        with sess.as_default():

            sess.run(init)

            for e in range(epochs):

                # Shuffle the whole data ahead of this epoch

                shuffled_idx = np.random.permutation(len(all_xs))
                all_xs = all_xs[shuffled_idx]
                all_ys = all_ys[shuffled_idx]


                for i in range(steps):

                    startindex = i * batch_size
                    endindex = startindex + batch_size

                    xs = all_xs[startindex:endindex]

                    ys = all_ys[startindex:endindex]

                    feed = {model.x: xs, model.y_: ys}

                    # Later models might need to add extra placeholders during training
                    if hasattr(model, 'validate_feed_dict') and callable(getattr(model, 'validate_feed_dict')):
                        feed = model.validate_feed_dict(feed, training=True)

                    sess.run(model.optimize, feed_dict=feed)


                    # Print result to screen for every 1000 iterations
                    if (i + 1) % 1000 == 0:
                        print("After %d iteration:" % i)

                        print("Cost: %f" % sess.run(model.cost, feed_dict={model.x: all_xs, model.y_: all_ys}))

                print("End of epoch {0:d}. Cost: {1:f}".format(e, sess.run(model.cost, feed_dict={model.x: all_xs, model.y_: all_ys})))

            model.save()

            # Try on first row:
            row = [data[0][0:9]]

            print(row)
            print(sess.run(model.prediction, feed_dict={model.x: row}))

            print(model.do_move(row[0]))



train_model(int(len(data)/batch_size), batch_size, epochs)
