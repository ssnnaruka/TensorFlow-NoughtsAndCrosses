class Game(object):

    winning_rows = (
        [0,1,2],
        [3,4,5],
        [6,7,8],
        [0,3,6],
        [1,4,7],
        [2,5,8],
        [0,4,8],
        [2,4,6]
    )

    def __init__(self, player1, player2):
        self.player1 = player1
        self.player2 = player2
        self.nextplayer = 1
        self.board = [0]*9
        self.winner = 0
        self.gameover = False
        self.journal = []

    def advance(self):
        if self.gameover:
            return True

        p = self.player1 if self.nextplayer == 1 else self.player2

        m = p.do_move(self.perspective_board(list(self.board), self.nextplayer))

        # Save a copy of what happened
        # board, next move, which player made that move
        self.journal.append([list(self.board), m, self.nextplayer])

        # Make the move
        self.board[m] = self.nextplayer

        self.nextplayer = (self.nextplayer % 2) + 1

        if self.board.count(0) == 0:
            self.gameover = True

        self.winner = self.get_winner()

        self.gameover = self.winner or self.gameover

        return self.gameover

    def get_winner(self):
        for comb in self.winning_rows:
            if self.board[comb[0]] == self.board[comb[1]] and self.board[comb[1]] == self.board[comb[2]] and self.board[comb[0]] != 0:
                return self.board[comb[0]]
        return 0

    def output(self):
        def c(i):
            if i == 0:
                return ' '
            return chr(i + ord('0'))

        print(" %c | %c | %c     0 1 2" % (c(self.board[0]), c(self.board[1]), c(self.board[2])))
        print("-----------")
        print(" %c | %c | %c     3 4 5" % (c(self.board[3]), c(self.board[4]), c(self.board[5])))
        print("-----------")
        print(" %c | %c | %c     6 7 8" % (c(self.board[6]), c(self.board[7]), c(self.board[8])))
        print("")

    @staticmethod
    def perspective_board(b, playernumber):
        for i in range(0, len(b)):
            if b[i] != 0:
                if b[i] == playernumber:
                    b[i] = 1
                else:
                    b[i] = -1
        return b

    @staticmethod
    def perspective_winner(winner, playernumber):
        if winner != 0:
            if winner == playernumber:
                return 1
            else:
                return -1
        return 0

    def get_journal(self):
        # Get a finalized journal
        # board (-1,0,1 -> where 1 is the player who is about to move)
        # Move they then made
        # Did they win in the end -> 1 for win, 0 for draw, -1 for lose

        final_journal = []

        for (b, m, p) in self.journal:
            b = self.perspective_board(b, p)

            w = self.perspective_winner(self.winner, p)

            final_journal.append([b,m,w])

        return final_journal
