import random


class Player(object):

    def name(self):
        return self.__class__.__name__

    def do_move(self, board):
        raise ValueError('Player is an abstract class')

    @staticmethod
    def get_free_cells(board):
        free_cells = []

        for i in range(0, 9):
            if board[i] == 0:
                free_cells.append(i)

        return free_cells


class RandomPlayer(Player):

    def do_move(self, board):
        free_cells = self.get_free_cells(board)

        return random.choice(free_cells)


class HumanPlayer(Player):

    def do_move(self, board):
        free_cells = self.get_free_cells(board)

        cell = input("Your move 0-8: ")

        return int(cell)


class SequentialPlayer(Player):

    def do_move(self, board):
        free_cells = self.get_free_cells(board)
        return free_cells[0]

