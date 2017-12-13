# Noughts and Crosses
## Introduction
Most tutorials for machine learning ultimately end up in a script that outputs a number such as accuracy that might increase from 60% as a baseline to 92% after applying an ML algorithm. The training and evaluation processes are both part of the same script.
It's just not very inspiring – I wanted to build something using ML that I could interact with, and feel how the ML techniques have actually contributed to it.
After following a few tutorials (MNIST etc), I had a reasonable feel for Tensorflow and the Python environment needed for ML projects. Given a well-defined problem and the instruction to apply a specific algorithm, I felt I could probably work out how to get Tensorflow to do that. But I still had no idea which algorithms I would choose for a new project of my own. Take Deep MNIST for example: the multiple convultional/pooled layers sound reasonable, but there is just no way you would automatically decide that the 'right' network would be the one you are instructed to build in that tutorial. The problem (most likely) is that it is just a question of trial and error, and experience. I suspect that even understanding the algorithms at a theoretical level would barely help…
So I wanted a 'playground' project to experiment with the available techniques.
Inspired by AlphaGo Zero, I decided to build a Noughts and Crosses (Tic Tac Toe) playground, to see where that takes me. Crucially, if I can build a Machine Intelligence robot to play the game, I can get it to play against myself (and itself…)
The game is not really very interesting mathematically. The player who starts has a significant advantage (and I believe can even always force a draw or a win). It also has a small 'universe'. Each square can have one of three values (blank, O or X), and there are only three squares. So the total possible space is 3^9 = 19,683 possible combinations – and this is further reduced if you apply the constraint that there can be at most one more square owned by player one compared to player two.  From a practical point of view, you could reduce this even further if you consider rotations and mirror images to represent the same basic game play.
So almost certainly, a search-based approach would be sufficient to tackle this problem, to find the next move that minimises the likelihood of losing.

## Setup – the non-ML framework
Game.py and player.py. Game is instantiated with two Player objects – e g.  a RandomPlayer and a HumanPlayer. The RandomPlayer just randomly places its piece in an available spot. The HumanPlayer asks the computer operator which square to use. You can use main.py to run two players against each other in one game – just pick which class of Player you assign to p1 or p2.
Within Game, the board is internally represented as a 1-dim array of 9 elements. Each element may be 0 for blank, 1 for player one's piece, and 2 for player two's. (We don't bother assigning O or X – they just stay mostly as 1 or 2.)
Game's advance method will cause it to ask the next player to make a move (it calls player.do_move(board). In this case, board is adjusted so the player's one pieces appear as 1's and the opponents as -1's. It will be inverted when shown to the other player – so player two also sees its own pieces as 1's (and player one's as -1's). The advance method returns True if the game is over, False if there is still another move to be made. Once over, Game's get_winner function returns 1 or 2 to indicate which player won, or 0 for a draw.
Game has an output method to display the current state of the board at any time. Once the game is over, there is a get_journal method on Game to return each stage of the game in an array. This can be used to analyse the game at each stage now that we know the ultimate winner (was it a good move or not?).
Generating Games
We now want to do two things – build a framework for playing Players against each other, and also generate some random game data that we might be able to use to build our machine intelligence. 
Look at generate_games.py. In p1 and p2, make sure it is creating RandomPlayers. The script will run as many games as specified in the games variable (e.g. 1000) and then output the scores to the screen. For example, with RandomPlayer versus RandomPlayer, you will probably see numbers like these:
```
Player 1 (RandomPlayer): Won 579
Player 2 (RandomPlayer): Won 282
Draws: 139
```
This shows the great advantage of being allowed to make the first move! (Player 1 is always asked to go first by the Game object.)

The generate_games.py script also outputs every stage of the game to a file (e.g. data/Match-RandomPlayer-RandomPlayer-1000.csv). This is essentially a flattened version of each get_journal() from the Game objects.
The format is as follows, with no header row:
```Columns 0-8, Next Move, Won Ultimately```
Each of the columns 0-8 is from the perspective of the player who is about to make the next move. So their own pieces appear as 1's, opponent's as -1's, blanks as 0's. Next Move gives the cell number (0-8) of the move the player decided to make after reviewing the board presented here (Columns 0-8). Then Won Ultimately tells us whether the player ended up winning (1), losing (-1), or drawing (0).

So if we run RandomPlayer against RandomPlayer 1,000 or certainly 10,000 games, we should have quite a lot of game play data (including multiple board positions for each game).
First Machine Learning Algorithm


https://danijar.com/structuring-your-tensorflow-models/

File structure
https://blog.metaflow.fr/tensorflow-a-proposal-of-good-practices-for-files-folders-and-models-architecture-f23171501ae3



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












