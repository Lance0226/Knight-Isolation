"""This file contains a collection of player classes for comparison with your
own agent and example heuristic functions.

    ************************************************************************
    ***********  YOU DO NOT NEED TO MODIFY ANYTHING IN THIS FILE  **********
    ************************************************************************
"""

from random import randint, random
import numpy as np
import random
from isolation import Board

TIME_LIMIT_MILLIS = 150
ACTION_TOTAL = 8
STATE_TOTAL = (7 * 7) ** 2
QTABLE = np.zeros((STATE_TOTAL, ACTION_TOTAL))


def null_score(game, player):
    """This heuristic presumes no knowledge for non-terminal states, and
    returns the same uninformative value for all other states.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state.
    """

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    return 0.


def open_move_score(game, player):
    """The basic evaluation function described in lecture that outputs a score
    equal to the number of moves open for your computer player on the board.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    return float(len(game.get_legal_moves(player)))


def improved_score(game, player):
    """The "Improved" evaluation function discussed in lecture that outputs a
    score equal to the difference in the number of moves available to the
    two players.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves - opp_moves)


def center_score(game, player):
    """Outputs a score equal to square of the distance from the center of the
    board to the position of the player.

    This heuristic is only used by the autograder for testing.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    w, h = game.width / 2., game.height / 2.
    y, x = game.get_player_location(player)
    return float((h - y) ** 2 + (w - x) ** 2)


class RandomPlayer():
    """Player that chooses a move randomly."""

    def get_move(self, game, time_left):
        """Randomly select a move from the available legal moves.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        ----------
        (int, int)
            A randomly selected legal move; may return (-1, -1) if there are
            no available legal moves.
        """
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return (-1, -1)
        return legal_moves[randint(0, len(legal_moves) - 1)]


class GreedyPlayer():
    """Player that chooses next move to maximize heuristic score. This is
    equivalent to a minimax search agent with a search depth of one.
    """

    def __init__(self, score_fn=open_move_score):
        self.score = score_fn

    def get_move(self, game, time_left):
        """Select the move from the available legal moves with the highest
        heuristic score.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        ----------
        (int, int)
            The move in the legal moves list with the highest heuristic score
            for the current game state; may return (-1, -1) if there are no
            legal moves.
        """
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return (-1, -1)
        _, move = max([(self.score(game.forecast_move(m), self), m) for m in legal_moves])
        return move


class HumanPlayer():
    """Player that chooses a move according to user's input."""

    def get_move(self, game, time_left):
        """
        Select a move from the available legal moves based on user input at the
        terminal.

        **********************************************************************
        NOTE: If testing with this player, remember to disable move timeout in
              the call to `Board.play()`.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        ----------
        (int, int)
            The move in the legal moves list selected by the user through the
            terminal prompt; automatically return (-1, -1) if there are no
            legal moves
        """
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return (-1, -1)

        print(game.to_string())  # display the board for the human player
        print(('\t'.join(['[%d] %s' % (i, str(move)) for i, move in enumerate(legal_moves)])))

        valid_choice = False
        while not valid_choice:
            try:
                index = int(input('Select move index:'))
                valid_choice = 0 <= index < len(legal_moves)

                if not valid_choice:
                    print('Illegal move! Try again.')

            except ValueError:
                print('Invalid index! Try again.')

        return legal_moves[index]


def state_to_index(state):  # 0，0，3，0
    """
    This assigns each states a unique index
    """
    factors = [7, 7, 7, 7]
    idx = 0
    coef = 1
    for i in range(4):
        idx += state[i] * coef
        coef *= factors[i]

    return int(idx)  # 240


def index_to_state(idx):
    """
    This method is the inverse of "state_to_index":
    Given an integer index, it reconstructs the corresponding state.
    """
    factors = [7, 7, 7, 7]
    state = []
    for i in range(4):
        digit = idx % factors[i]
        idx = (idx - digit) / factors[i]
        state.append(digit)
    return tuple(state)


def move_to_action(pos, mov):
    x = mov[0] - pos[0]
    y = mov[1] - pos[1]
    if x == 1:
        if y == 2:
            idx = 0
        if y == -2:
            idx = 3
    if x == 2:
        if y == 1:
            idx = 1
        if y == -1:
            idx = 2
    if x == -1:
        if y == 2:
            idx = 7
        if y == -2:
            idx = 4
    if x == -2:
        if y == 1:
            idx = 6
        if y == -1:
            idx = 5
    return idx


def action_to_move(pos, idx):
    x = pos[0]
    y = pos[1]
    if idx == 0:
        return x + 1, y + 2
    if idx == 1:
        return x + 2, y + 1
    if idx == 2:
        return x + 2, y - 1
    if idx == 3:
        return x + 1, y - 2
    if idx == 4:
        return x - 1, y - 2
    if idx == 5:
        return x - 2, y - 1
    if idx == 6:
        return x - 2, y + 1
    if idx == 7:
        return x - 1, y + 2


class QLearningPlayer():

    def __init__(self, exp_exp_tradeoff, epsilon, state):
        self.exp_exp_tradeoff = exp_exp_tradeoff
        self.epsilon = epsilon
        self.state = state

    def get_move(self, game, time_left):
        if self.exp_exp_tradeoff > self.epsilon:  # Exploit
            action = np.argmax(QTABLE[self.state, :])
        else:  # Explore
            legal_moves = game.get_legal_moves()
            if not legal_moves:
                action = (-1, -1)
            action = legal_moves[randint(0, len(legal_moves) - 1)]


# Yan Gong
def ex1(num_iters):
    from isolation import Board

    #num_iters = 10000  # Total number of episodes
    max_steps = 100  # Max steps per episode ??
    a = 0.8  # Learning rate
    y = 0.9  # Discounting rate
    epsilon = 1.0  # Exploration rate
    max_epsilon = 1.0  # Exploration Max
    min_epsilon = 0.01  # Minimum Min
    decay_rate = 0.005  # Decay Rate

    win=0
    rewards = []
    for i in range(num_iters):
        stateI = (0, 0, 0, 6)
        state = state_to_index(stateI)
        exp_exp_tradeoff = random.uniform(0, 1)
        total_rewards = 0

        player1 = QLearningPlayer(exp_exp_tradeoff, epsilon, state)  # Initial for a new game
        player2 = RandomPlayer()
        game = Board(player1, player2)
        game.apply_move((0, 0))
        game.apply_move((0, 6))

        for j in range(max_steps):                                      # One game, break when someone win or lose
            pos = game.get_player_location(player1)
            pos2 = game.get_player_location(player1)

            legal_moves = game.get_legal_moves(player1)                 # Q-Learning move for player1
            if exp_exp_tradeoff > epsilon:  # Exploit
                i = 0
                maxindex = 0
                for tmp in QTABLE[state, :]:
                    if tmp > QTABLE[state, maxindex] and game.move_is_legal(action_to_move(pos, i)):
                        maxindex = i
                i += 1
                action = maxindex

            else:  # Explore
                move = legal_moves[randint(0, len(legal_moves) - 1)]
                action = move_to_action(pos, move)

            game.apply_move(move)

            if game.is_winner(player1) or game.is_loser(player1):
                if(game.is_winner(player1)):
                    win=win+1
                break

            legal_moves2 = game.get_legal_moves(player2)        # Random move for player2
            move2 = legal_moves2[randint(0, len(legal_moves2) - 1)]
            action2 = action_to_move(pos2, move2)
            game.apply_move(move2)
            location1 = game.get_player_location(player1)
            location2 = game.get_player_location(player2)

            new_state = state_to_index(location1 + location2)
            reward = game.reward(player1)

            QTABLE[state, action] = QTABLE[state, action] + a * (  # Update Q Table
                    reward + y * np.max(QTABLE[new_state, :]) - QTABLE[state, action])

            total_rewards += reward
            state = new_state

            if game.is_winner(player1) or game.is_loser(player1):
                if (game.is_winner(player1)):
                    win = win + 1
                break

        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * i)
        rewards.append(total_rewards)

    print("Win Rate")
    print(win/num_iters)


if __name__ == "__main__":
    from isolation import Board

    print("Training 100 Times")
    ex1(100)  # Training 100 Times
    print("Training 200 Times")
    ex1(100)  # Training 200 Times
    print("Training 1000 Times")
    ex1(1000)         # Training 1000 Times
    print("Training 2000 Times")
    ex1(2000)  # Training 1000 Times
    print("Training 10000 Times")
    ex1(10000)         # Training 10000 Times
