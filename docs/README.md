# Noughts and Crosses
## Introduction
Most tutorials for machine learning ultimately end up in a script that outputs a number such as accuracy that might increase from 60% as a baseline to 92% after applying an ML algorithm. The training and evaluation processes are both part of the same script.
It's just not very inspiring – I wanted to build something using ML that I could interact with, and feel how the ML techniques have actually contributed to it.

After following a few tutorials ([MNIST](https://www.tensorflow.org/get_started/mnist/beginners) etc), I had a reasonable feel for Tensorflow and the Python environment needed for ML projects. Given a well-defined problem and the instruction to apply a specific algorithm, I felt I could probably work out how to get Tensorflow to do that. But I still had no idea which algorithms I would choose for a new project of my own. Take [Deep MNIST](https://www.tensorflow.org/get_started/mnist/pros) for example: the multiple convultional/pooled layers sound reasonable, but there is just no way you would automatically decide that the 'right' network would be the one you are instructed to build in that tutorial. The problem (most likely) is that it is just a question of trial and error, and experience. I suspect that even understanding the algorithms at a theoretical level would barely help...

So I wanted a 'playground' project to experiment with the available techniques.

Inspired by AlphaGo Zero, I decided to build a Noughts and Crosses (Tic Tac Toe) playground, to see where that takes me. Crucially, if I can build a Machine Intelligence robot to play the game, I can get it to play against myself (and itself...)

**This tutorial will explain the project and lead you through generating data to train various Machine Learning models, allowing you to play against the Machine Intelligence you've built. It assumes you have already installed TensorFlow on your machine and followed some of the basic 'deep learning' examples. The journey presented is my own experience experimenting with TensorFlow to understand how to use it.**

The game Noughts and Crosses is not really very interesting mathematically. The player who starts has a significant advantage (and I believe can even always force a draw or a win). It also has a small 'universe'. Each square can have one of three values (blank, O or X), and there are only nine squares. So the total possible space is 3^9 = 19,683 possible combinations – and this is further reduced if you apply the constraint that there can be at most one more square owned by player one compared to player two.  From a practical point of view, you could reduce this even further if you consider rotations and mirror images to represent the same basic game play.

So almost certainly, a search-based approach would be sufficient to tackle this problem, to find the next move that minimises the likelihood of losing.

## Setup – the non-ML framework

The files game.py and player.py contain the basic classes that allow the game to be played, in a flexible structure that means we can inherit classes to add Machine Learning versions of the player later on. Game is instantiated with two Player objects – e g.  a RandomPlayer and a HumanPlayer. The RandomPlayer just randomly places its piece in an available spot. The HumanPlayer asks the computer operator which square to use. You can use main.py to run two players against each other in one game – just pick which class of Player you assign to p1 or p2.

For example, to pit the RandomPlayer against yourself (HumanPlayer) with the RandomPlayer as player 1 and making the first move, run the following from the root project directory having cloned the GitHub repo to your computer (you may need to adjust depending on how Python 3 is installed on your computer):

```
python3 main.py RandomPlayer HumanPlayer
```

The computer will place its first '1' piece in a random square, and then you are presented with the current state of the board as well as a reminder (on the right hand side) of the numbers 0-8 that you can enter to choose a square for your move.

Within Game, the board is internally represented as a 1-dim array of 9 elements. Each element may be 0 for blank, 1 for player one's piece, and 2 for player two's. (We don't bother assigning O or X – they just stay as 1 or 2.)

Game's advance method will cause it to ask the next player to make a move (it calls player.do_move(board). In this case, board is adjusted so the player's one pieces appear as 1's and the opponents as -1's. It will be inverted when shown to the other player – so player two also sees its own pieces as 1's (and player one's as -1's). The advance method returns True if the game is over, False if there is still another move to be made. Once over, Game's get_winner function returns 1 or 2 to indicate which player won, or 0 for a draw.

Game has an output method to display the current state of the board at any time. Once the game is over, there is a get_journal method on Game to return each stage of the game in an array. This can be used to analyse the game at each stage now that we know the ultimate winner (was it a good move or not?).

### Generating Games
We now want to do two things - build a framework for playing Players against each other, and also generate some random game data that we might be able to use to build our machine intelligence. That's what AlphaGo Zero supposedly did to learn the how to play Go!

Look at generate_games.py. You can tell it to play 1000 games of RandomPlayer versus RandomPlayer as follows:

```
python3 generate_games.py RandomPlayer RandomPlayer --games 1000
```

The script will run as many games as specified in the games variable (e.g. 1000) and then output the final scores to the screen. For example, with RandomPlayer versus RandomPlayer, you will probably see numbers like these:

```
Player 1 (RandomPlayer): Won 579
Player 2 (RandomPlayer): Won 282
Draws: 139
```

This shows the great advantage of being allowed to make the first move! (Player 1 is always asked to go first by the Game object.)

The generate_games.py script also outputs every stage of every game to a file (e.g. data/Match-RandomPlayer-RandomPlayer-1000.csv). This is essentially a flattened version of each get_journal() from the Game objects. For a finished Game, get_journal() will return an array showing the game play at each stage of the game, including knowledge of the move the next player made and whether this ultimate led to a win. This information could help our ML player learn the best moves to make given the same board layouts during its own games.

The format of the file is as follows, with no header row: 
```Column 0, Col 1, Col 2, ..., Col 8, Next Move, Won Ultimately```

Each of the columns 0-8 is from the perspective of the player who is about to make the next move. So their own pieces appear as 1's, opponent's as -1's, blanks as 0's. Next Move gives the cell number (0-8) of the move the player decided to make after reviewing the board presented here (Columns 0-8). Then Won Ultimately tells us whether the player ended up winning (1), losing (-1), or drawing (0).

So if we run RandomPlayer against RandomPlayer 1,000 or certainly 10,000 games, we should have quite a lot of game play data (including multiple board positions for each game).

If you want to get a feel for matching different types of players against each other, without having to act as a HumanPlayer for 100s of games, try using SequentialPlayer. For example:

```
python3 generate_games.py RandomPlayer SequentialPlayer --games 1000
```

SequentialPlayer is another simple algorithm for playing the game, but without randomness - it just picks the first free square it can find (looking from 0-8) and chooses to place its piece there.

What happens if you switch SequentialPlayer to go as player 1, and RandomPlayer as player 2? What about if you pit SequentialPlayer against itself?

## First Machine Learning Algorithm

Now we have a great framework for working with the Noughts and Crosses game, and even generated some random gameplay data that we believe could be used to train a Machine Learning model, so it's time to see if we can build a ML-based Player class!

Previous examples I'd looked at did the following all in one main script: build a TensorFlow graph, load in some training and test data, run some kind of optimization step to train the weights in the model, then evaluate the model compared to the test data. This is the 'boring stats' approach of the basic tutorials that I want to move away from! Don't just let the model die once we've got an accuracy score of some kind.

We want to build a basic TrainedPlayer class, inheriting from Player, which must be used in two distinct ways:

1. Train the model based on some of our 'generate_games.py' data that we produced earlier, then save the model weights.
2. Load the trained model so that it is immediately available to play a game or two of Noughts and Crosses, potentially against another instance of itself.

There are some important TensorFlow concepts that I had to discover here. 

How to create a Python class representation of a model so that we can plug the same graph into training and evaluation modes, accessed in different ways, without repeating code - the [Structuring Your TensorFlow Models](https://danijar.com/structuring-your-tensorflow-models/) article by Danijar Hafner is very helpful here.

Take a look at trained_player.py. The decorator lazy_property is taken from these ideas, and in the class TrainedPlayer you can see prediction, cost, and optimize properties (plus accuracy) that are the 'graph components' to be used in training and reused in evaluation. These can also be extended if we inherit the base TrainedPlayer class to support a different model.

Another useful article is [TensorFlow: A proposal of good practices for files, folders and models architecture](https://blog.metaflow.fr/tensorflow-a-proposal-of-good-practices-for-files-folders-and-models-architecture-f23171501ae3) by Morgan Giraud. We will borrow some ideas for using the TensorFlow [Saver](https://www.tensorflow.org/programmers_guide/saved_model) class to save and restore models and data, as well as having our model object maintain its own [Graph](https://www.tensorflow.org/api_docs/python/tf/Graph) object so that it can evaluate its model without conflicting with the default TensorFlow graph - which would otherwise happen if we try to run two ML Players at the same time since the second instance would try to redefine the same variables in TensorFlow.

### Basic Neural Network model

Let's think about the way we want the model to be used ultimately: given an array of nine current board positions, return the best square to pick for our next move.

The input will be a 1-dim array with nine elements (each -1, 0, or 1) showing the current board state from this player's perspective (as defined in the description of the CSV files for historical gameplay data, above). So you might feed `[0, 1, 0, -1, 1, -1, 0, 0, 0]` as a board. This means that the current player has its pieces in squares 1 and 4, and the opponent in squares 3 and 5. (Squares are numbered 0-8 from top left of the board, running horizontally first, then the next row etc.) In this example, the best move is clearly to put our next piece (a 1) in square 7 to complete the middle vertical and win the game.

So the output will just be a number such as 7 indicating the move that our model thinks we should make.

However, in general there is no absolute right answer - just better or worse answers.

You may remember from the MNIST tutorials that we actually end up with 10 probabilities showing the likelihood of each of the digits 0-9 being the digit that the handwritten input was supposed to represent. So let's borrow this idea, and in fact come up with a model that outputs a 1-dim array of nine elements, each representing the probability of that square being the best move.

In `__init__` of TrainedPlayer, the input and _real-world_ (as opposed to model-predicted) outputs are represented as follows:

```
self.x = tf.placeholder(tf.float32, [None, 9], name="x")
self.y_ = tf.placeholder(tf.float32, [None, 9], name="y_")
```

The `[None, 9]` means a 2-dim matrix of an unspecified number of rows, each of length 9. In training, we will supply multiple rows to the graph in one go; in evaluation we only supply one, representing the current board state.

The basic TrainedPlayer has a prediction property representing a single 'perceptron' i.e. just one linear node of a neural network: y = Wx + b for weights W and constant bias term. How did I know to try that? It's the basic Deep MNIST solution.

Likewise, I took the `optimize` property to be `tf.train.GradientDescentOptimizer(0.5).minimize(self.cost)` based on simple examples, where `self.cost` is `cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.prediction))` - that's what we want to minimze. The softmax activation function is the one that outputs a tensor of probabilities.

### Training the Model

See the file train.py to try training our basic model. We need to choose a CSV file of gamedata to process, and parameters batchsize and epochs to tell it how many times to iterate through a full training loop (epochs) and how many board-position/winning-move pairs to try to optimize against in each step (batchsize). Again, the gameplay data was generated earlier, above, so provide a run of RandomPlayer versus itself games.

```
python3 train.py TrainedPlayer -i 'data/Match-RandomPlayer-RandomPlayer-100000.csv' --batchsize 10 --epochs 10
```

The train.py script loads the CSV and filters all the data so that it only keeps board positions where the corresponding move ultimately led to a win. There might be a way to train it _not_ to make a move that lost, but for now let's just focus on the 'winning' data. Remember the CSV has 11 columns: 0-8 represent a board position (current player represented by 1s), column 9 gives the move that was made next (0-8), and column 10 says which player ultimately won the game (1 for the current player, 0 for a draw, -1 if the other player won). So we just keep rows where column 10 is equal to 1.

The script translates the remaining data into: all_xs - an array of all the current board positions only (i.e. just columns 0-8); and all_ys - an array of the winning moves in 'one-hot form', that is rows of 9 elements showing a 1 only in the column corresponding to the move, so a move in square 7 is represented by `[0, 0, 0, 0, 0, 0, 0, 1, 0]`.

Iterating through each epoch and batch, the script basically just calls `sess.run(model.optimize)` on the current batch subset taken from all_xs and all_ys. Notice how it wraps everything in `with model.graph.as_default()` to ensure it accesses the TensorFlow graph specific to the TrainedPlayer object. It wouldn't matter here if everything was actually just defined on the default graph, but this could conflict at the evaluation stage if multiple graphs are loaded by different TrainedPlayers.

After training, it calls `model.save()` to output ckpt file(s) to the models folder so the weights can be loaded back in future. It also prints a prediction on the first board available in the CSV just so you can see a prediction in action.

Basically, all we've done is find a nice structured way to run something like this example TensorFlow code, except iterated over in epochs/batches, and then saved at the end:

```
x = tf.placeholder(tf.float32, [None, 9], name="x")
y_ = tf.placeholder(tf.float32, [None, 9], name="y_")

W = tf.Variable(tf.zeros([9, 9]), name="W")
b = tf.Variable(tf.zeros([9]), name="b")

y = tf.nn.softmax(tf.matmul(x, W) + b, name="prediction")

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()

with tf.Session() as sess:
   sess.run(init)
   sess.run(train_step, feed_dict={model.x: xs, model.y_: ys})
```



Deep Learning 1

Deep2TrainedPlayer
```
p2 \ p1              RandomPlayer TrainedPlayer Deep2TrainedPlayer
RandomPlayer              51 (17)        92 (2)             85 (3)
```
Hidden_nodes = 20
This is with 100,000 training games and batch_size of 100. Batch_size 10 gave 79.

Batch_size 1000 gives 82.

100000 games dataset / batch_size 100 -> 899

With 1 epoch -> cost lowers to 2.179088
With 10 epochs -> cost lowers to 2.038518, Score against RandomPlayer is 930 (maybe that was with 20 hidden nodes?)
10 epochs, batchsize 1000 -> cost 2.178687, Score 906

I read that a discontinuous function requires at least 2 hidden layers to be approximated.

Deep2HiddenTrainedPlayer:
Batchsize 100, epochs 1, cost to 2.188, score 837	
Batchsize 1000, epochs 1, cost to 2.195, score 850 (not sure why higher)

Batchsize 100, epochs 100, cost didn't seem to go below the 10 epoch seen above. Maybe we only had 10 neurons…?

20 Neurons on 2 hidden layers each
Batchsize 100, epochs 1, cost to 2.18

Batchsize 10, epochs 1, cost to 2.04, Score 951

First time we've seen better than TrainedPlayer against RandomPlayer!

How about 50 neurons?
Batchsize 10, epochs 1, cost to 2.04, Score 941

So around the same again

Maybe more epochs would help, or increased batch size so the neurons can pick up more or something…?

Batchsize 100, epochs 5, cost to 2.04, score still a bit lower

How about 100 neurons?
Batchsize 10, epochs 10. Cost to 2.004796, score 980.

Also inspiring that it beats both itself and TrainedPlayer every time (and TrainedPlayer even if it goes first). But these are deterministic of course, so not actually significant.
Playing against it, it always seems to at least make the winning move if there is one available. It's obvious to us, and the first thing a human-programmed solution would do is to see if there is a single winning move. But amazing to think that's emerged statistically.


Back to one hidden layer (Deep2TrainedPlayer).What about hidden_nodes = 100?
pythonw train.py Deep2TrainedPlayer -i 'data/Match-RandomPlayer-RandomPlayer-100000.csv' --batchsize 10 --epochs 10
Cost 2.012, Score 964
So, better than 2 layers with 50 neurons, but not as good as 2 layers with 100 neurons.

How about 200 neurons in the hidden layer?
Cost 2.001493, Score 995


How about 500 neurons in the hidden layer?
Cost 2.010671, Score 974












