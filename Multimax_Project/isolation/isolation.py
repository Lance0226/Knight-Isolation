"""
This file contains the `Board` class, which implements the rules for the
game Isolation as described in lecture, modified so that the players move
like knights in chess rather than queens.

You MAY use and modify this class, however ALL function signatures must
remain compatible with the defaults provided, and none of your changes will
be available to project reviewers.
"""

'''
This project is revised by the open source code https://github.com/dingran/knight-isolation.

The author of modification is Lu Han lhan11@syr.edu

'''

import random
import timeit
import time
from copy import copy

TIME_LIMIT_MILLIS = 15000


class Board(object):
    """Implement a model for the game Isolation assuming each player moves like
    a knight in chess.

    Parameters
    ----------
    player_1 : object
        An object with a get_move() function. This is the only function
        directly called by the Board class for each player.

    player_2 : object
        An object with a get_move() function. This is the only function
        directly called by the Board class for each player.

    width : int (optional)
        The number of columns that the board should have.

    height : int (optional)
        The number of rows that the board should have.
    """
    BLANK = 0
    NOT_MOVED = None

    def __init__(self, player_1, player_2, player_3, width, height):
        #print("test test test")
        self.width = width
        self.height = height
        self.move_count = 0
        self._player_1 = player_1
        self._player_2 = player_2
        self._player_3 = player_3
        self._active_player = player_1
        self._inactive_players = [player_2,player_3]

        # The last 4 entries of the board state includes initiative (0 for
        # player 1, 1 for player 2, 2 1 for player 3) player 1 last move, player 2 last move and player 3 last move
        self._board_state = [Board.BLANK] * (width * height + 4)
        self._board_state[-1] = Board.NOT_MOVED
        self._board_state[-2] = Board.NOT_MOVED
        self._board_state[-3] = Board.NOT_MOVED


    def hash(self):
        return str(self._board_state).__hash__()

    @property
    def board_state(self):
        return self._board_state

    @property
    def active_player(self):
        """The object registered as the players list holding initiative in the
        current game state.
        """
        return self._active_player

    @property
    def inactive_players(self):
        """The object registered as the player in waiting for the current
        game state.
        """
        return self._inactive_players

    def copy(self):
        """ Return a deep copy of the current board. """
        #print "self._board_state"
        #print self._board_state;
        # time.sleep(1)
        new_board = Board(self._player_1, self._player_2, self._player_3, width=self.width, height=self.height)
        new_board.move_count = self.move_count
        new_board._active_player = self._active_player
        new_board._inactive_players = self._inactive_players
        new_board._board_state = copy(self._board_state)
        # print "self._board_state0"
        # print self._board_state;
        return new_board

    def forecast_move(self, move):
        """Return a deep copy of the current game with an input move applied to
        advance the game one ply.

        Parameters
        ----------
        move : (int, int)
            A coordinate pair (row, column) indicating the next position for
            the active player on the board.

        Returns
        -------
        isolation.Board
            A deep copy of the board with the input move applied.
        """
        new_board = self.copy()
        new_board.apply_move(move)
        return new_board

    def move_is_legal(self, move):
        """Test whether a move is legal in the current game state.

        Parameters
        ----------
        move : (int, int)
            A coordinate pair (row, column) indicating the next position for
            the active player on the board.

        Returns
        -------
        bool
            Returns True if the move is legal, False otherwise
        """
        idx = move[0] + move[1] * self.height
        return (0 <= move[0] < self.height and 0 <= move[1] < self.width and
                self._board_state[idx] == Board.BLANK)

    def get_blank_spaces(self):
        """Return a list of the locations that are still available on the board.
        """
        return [(i, j) for j in range(self.width) for i in range(self.height)
                if self._board_state[i + j * self.height] == Board.BLANK]

    def get_player_location(self, player):
        """Find the current location of the specified player on the board.

        Parameters
        ----------
        player : object
            An object registered as a player in the current game.

        Returns
        -------
        (int, int) or None
            The coordinate pair (row, column) of the input player, or None
            if the player has not moved.
        """
        if player == self._player_1:
            if self._board_state[-3] == Board.NOT_MOVED:
                return Board.NOT_MOVED
            idx = self._board_state[-3]
        elif player == self._player_2:
            if self._board_state[-2] == Board.NOT_MOVED:
                return Board.NOT_MOVED
            idx = self._board_state[-2]
        elif player == self._player_3:
            if self._board_state[-1] == Board.NOT_MOVED:
                return Board.NOT_MOVED
            idx = self._board_state[-1]
        else:
            raise RuntimeError(
                "Invalid player in get_player_location: {}".format(player))
        w = idx // self.height
        h = idx % self.height
        return (h, w)

    def get_legal_moves(self, player=None):
        """Return the list of all legal moves for the specified player.

        Parameters
        ----------
        player : object (optional)
            An object registered as a player in the current game. If None,
            return the legal moves for the active player on the board.

        Returns
        -------
        list<(int, int)>
            The list of coordinate pairs (row, column) of all legal moves
            for the player constrained by the current game state.
        """
        if player is None:
            player = self.active_player
        return self.__get_moves(self.get_player_location(player))

    def apply_move(self, move):
        """Move the active player to a specified location.

        Parameters
        ----------
        move : (int, int)
            A coordinate pair (row, column) indicating the next position for
            the active player on the board.
        """
        # print "self._board_state-1"
        # print self._board_state;
        idx = move[0] + move[1] * self.height

        if self.active_player == self._player_1:
            last_move_idx = 3
        elif self.active_player == self._player_2:
            last_move_idx = 2
        else:
            last_move_idx = 1

        self._board_state[-last_move_idx] = idx

        if self._board_state[-4] == 0:
            self._board_state[-4] = 1
            self._board_state[idx] = 1
        elif self._board_state[-4] == 1:
            self._board_state[-4] = 2
            self._board_state[idx] = 2
        else:
            self._board_state[-4] = 0
            self._board_state[idx] = 3



      
        self._active_player, self._inactive_players = self._inactive_players[0], [self._inactive_players[1],self._active_player]
        # print  self._active_player, self._inactive_players 
        self.move_count += 1
        # print "self._board_state2"
        # print self._board_state;

    # def is_winner(self, player):
    #     """ Test whether the specified player has won the game. """
    #     return (player in self._inactive_players) and not self.get_legal_moves(self._active_player)

    def is_loser(self, player):
        """ Test whether the specified player has lost the game. """
        return not self.get_legal_moves(self._active_player)


    def __get_moves(self, loc):
        """Generate the list of possible moves for an L-shaped motion (like a
        knight in chess).
        """
        if loc == Board.NOT_MOVED:
            return self.get_blank_spaces()

        r, c = loc
        directions = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                      (1, -2), (1, 2), (2, -1), (2, 1)]
        valid_moves = [(r + dr, c + dc) for dr, dc in directions
                       if self.move_is_legal((r + dr, c + dc))]
        random.shuffle(valid_moves)
        return valid_moves

    def get_player(self):
        return self._board_state[-4]

    def to_string(self, symbols=['1', '2','3']):
        """Generate a string representation of the current game state, marking
        the location of each player and indicating which cells have been
        blocked, and which remain open.
        """
        

        p1_loc = self._board_state[-3]
        p2_loc = self._board_state[-2]
        p3_loc = self._board_state[-1]

        # print self._board_state,p1_loc,p2_loc,p3_loc

        col_margin = len(str(self.height - 1)) + 1
        prefix = "{:<" + "{}".format(col_margin) + "}"
        offset = " " * (col_margin + 3)
        out = offset + '   '.join(map(str, range(self.width))) + '\n\r'
        for i in range(self.height):
            out += prefix.format(i) + ' | '
            for j in range(self.width):
                idx = i * self.width + j 
                if not self._board_state[idx]:
                    out += ' '
                elif p1_loc == idx:
                    out += symbols[0]
                elif p2_loc == idx:
                    out += symbols[1]
                elif p3_loc == idx:
                    out += symbols[2]
                elif self._board_state[idx] == 1:
                    out += 'O'
                elif self._board_state[idx] == 2:
                    out += 'X'
                elif self._board_state[idx] == 3:
                    out += '+'
                out += ' | '
            out += '\n\r'

        return out

    def play(self, time_limit=TIME_LIMIT_MILLIS):
        """Execute a match between the players by alternately soliciting them
        to select a move and applying it in the game.

        Parameters
        ----------
        time_limit : numeric (optional)
            The maximum number of milliseconds to allow before timeout
            during each turn.

        Returns
        ----------
        ([player,player], list<[(int, int),]>, str)
            Return multiple including the winning player, the complete game
            move history, and a string indicating the reason for losing
            (e.g., timeout or invalid move).
        """
        move_history = []

        time_millis = lambda: 10000 * timeit.default_timer()

        while True:

   

            legal_player_moves = self.get_legal_moves()
            game_copy = self.copy()

            move_start = time_millis()
            time_left = lambda : time_limit - (time_millis() - move_start)
            curr_move = self._active_player.get_move(game_copy, time_left)
            move_end = time_left()

            if curr_move is None:
                curr_move = Board.NOT_MOVED

            if move_end < 0:
                return self._inactive_players, move_history, "timeout"

            if curr_move not in legal_player_moves:
                if len(legal_player_moves) > 0:
                    return self._inactive_players, move_history, "forfeit"
                return self._inactive_players, move_history, "illegal move"

            move_history.append(list(curr_move))

            self.apply_move(curr_move)
