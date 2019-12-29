"""This file contains a collection of player classes for comparison with your
own agent and example heuristic functions.

    ************************************************************************
    ***********  YOU DO NOT NEED TO MODIFY ANYTHING IN THIS FILE  **********
    ************************************************************************
"""

from random import randint, random
import numpy as np
import random

TIME_LIMIT_MILLIS = 150
ACTION_TOTAL = 8
BOARD_HEIGHT = 5
BOARD_WIDTH = 5
CURRENT_POSITION = BOARD_HEIGHT * BOARD_WIDTH
STATE_TOTAL = CURRENT_POSITION * 2 ** (BOARD_HEIGHT * BOARD_WIDTH + 1)
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

# Yan Gong
def position_to_index(state):  # 0，0，3，0
    return int(state[0] * BOARD_WIDTH + state[1])

# Yan Gong
def state_to_index(pos, state):
    idx = 0
    for (i, j) in state:
        idx = idx + 2 ** (i * BOARD_WIDTH + j)
    posIdx = int(pos[0] * BOARD_WIDTH + pos[1])
    idx = idx + posIdx * (2 ** (BOARD_WIDTH * BOARD_HEIGHT + 1))
    return int(idx)

# Yan Gong
def move_to_action(pos, mov):
    y = mov[0] - pos[0]
    x = mov[1] - pos[1]
    if x == 1:
        if y == -2:
            idx = 0
        if y == 2:
            idx = 3
    if x == 2:
        if y == -1:
            idx = 1
        if y == 1:
            idx = 2
    if x == -1:
        if y == -2:
            idx = 7
        if y == 2:
            idx = 4
    if x == -2:
        if y == -1:
            idx = 6
        if y == 1:
            idx = 5
    return idx

# Yan Gong
def action_to_move(pos, idx):
    x = pos[0]
    y = pos[1]
    if idx == 0:
        return x - 2, y + 1
    if idx == 1:
        return x - 1, y + 2
    if idx == 2:
        return x + 1, y + 2
    if idx == 3:
        return x + 2, y + 1
    if idx == 4:
        return x + 2, y - 1
    if idx == 5:
        return x + 1, y - 2
    if idx == 6:
        return x - 1, y - 2
    if idx == 7:
        return x - 2, y - 1

# Yan Gong
class QLearningPlayer():
    def get_move(self, game, time_left):
        pos = game.get_player_location(self)
        stateIdx = state_to_index(pos, game.get_occupied_spaces())
        legal_moves = game.get_legal_moves(self)
        i = 0
        maxindex = -1
        for tmp in QTABLE[stateIdx, :]:
            if tmp > QTABLE[stateIdx, maxindex] and game.move_is_legal(action_to_move(pos, i)):
                maxindex = i
        i += 1
        if (maxindex == -1):
            if not legal_moves:
                return (-1, -1)
            moves = legal_moves[randint(0, len(legal_moves) - 1)]
        else:
            moves = action_to_move(pos, maxindex)
        return moves


# Yan Gong
def ex1(num_iters):
    #num_iters = 1  # Total number of games
    max_steps = 100  # Max steps in one game
    a = 0.8  # Learning rate
    y = 0.9  # Discounting rate
    epsilon = 1.0  # Exploration rate
    epsilon_max = 1.0  # Exploration Max
    epsilon_min = 0.01  # Exploration Min
    decay = 0.01  # Decay rate

    win = 0
    win2 = 0
    win3 = 0
    rewards = []
    for i in range(num_iters):
        tradeoff = random.uniform(0, 1)
        total_rewards = 0

        player1 = QLearningPlayer()  # Initial for a new game
        player2 = RandomPlayer()
        player3 = RandomPlayer()
        game = Board(player1, player2, player3)

        game.apply_move((0,0))
        game.apply_move((1,1))
        game.apply_move((2,2))

        stateIdx = state_to_index(game.get_player_location(player1), game.get_occupied_spaces())

        for j in range(max_steps):  # One game, break when someone win or lose
            pos = game.get_player_location(player1)
            pos2 = game.get_player_location(player2)
            pos3 = game.get_player_location(player3)

            legal_moves = game.get_legal_moves(player1)  # Q-Learning move for player1
            if tradeoff > epsilon:  # Exploit
                i = 0
                maxindex = -1
                for tmp in QTABLE[stateIdx, :]:
                    if tmp > QTABLE[stateIdx, maxindex] and game.move_is_legal(action_to_move(pos, i)):
                        maxindex = i
                i += 1
                if (maxindex == -1):
                    if not legal_moves:
                        return (-1, -1)
                    move = legal_moves[randint(0, len(legal_moves) - 1)]
                else:
                    move = action_to_move(pos, maxindex)

            else:  # Explore
                move = legal_moves[randint(0, len(legal_moves) - 1)]
                action = move_to_action(pos, move)

            game.apply_move(move)

            if game.is_loser():
                win = win + 1
                win3 = win3 + 1
                reward = 100
                QTABLE[stateIdx, action] = QTABLE[stateIdx, action] + a * (  # Update Q Table
                        reward + y * np.max(QTABLE[stateIdx, :]) - QTABLE[stateIdx, action])
                break

            legal_moves2 = game.get_legal_moves(player2)  # Random move for player2
            move2 = legal_moves2[randint(0, len(legal_moves2) - 1)]
            action2 = action_to_move(pos2, move2)
            game.apply_move(move2)

            if game.is_loser():
                win = win + 1
                win2 = win2 + 1
                reward = 100
                QTABLE[stateIdx, action] = QTABLE[stateIdx, action] + a * (  # Update Q Table
                        reward + y * np.max(QTABLE[stateIdx, :]) - QTABLE[stateIdx, action])
                break

            legal_moves3 = game.get_legal_moves(player3)  # Random move for player3
            move3 = legal_moves3[randint(0, len(legal_moves3) - 1)]
            action3 = action_to_move(pos3, move3)
            game.apply_move(move3)

            if game.is_loser():
                win2 = win2 + 1
                win3 = win3 + 1
                reward = -100
                QTABLE[stateIdx, action] = QTABLE[stateIdx, action] + a * (  # Update Q Table
                        reward + y * np.max(QTABLE[stateIdx, :]) - QTABLE[stateIdx, action])
                break

            location1 = game.get_player_location(player1)

            new_state = state_to_index(location1, game.get_occupied_spaces())
            reward = game.reward(player1)

            QTABLE[stateIdx, action] = QTABLE[stateIdx, action] + a * (  # Update Q Table
                    reward + y * np.max(QTABLE[stateIdx, :]) - QTABLE[stateIdx, action])

            total_rewards += reward
            stateIdx = new_state

        epsilon = epsilon_min + (epsilon_max - epsilon_min) * np.exp(-decay * i)
        rewards.append(total_rewards)



if __name__ == "__main__":
    from isolation import Board

    win = 0;        # Random way
    for i in range(10000):
        player1 = RandomPlayer()
        player2 = RandomPlayer()
        player3 = RandomPlayer()
        game = Board(player1, player2, player3)
        game.apply_move((0, 0))
        game.apply_move((1, 1))
        game.apply_move((2, 2))
        winner, history, outcome = game.play()
        if (winner[0] == player1 or winner[1] == player1):
            win = win + 1
    print("Use Random Way, the win rate for player1 is: ")
    print(win / 10000)

    ex1(10)        # Training for 100 times
    win = 0;
    for i in range(10000):
        player1 = QLearningPlayer()
        player2 = RandomPlayer()
        player3 = RandomPlayer()
        game = Board(player1, player2, player3)
        game.apply_move((0, 0))
        game.apply_move((1, 1))
        game.apply_move((2, 2))
        winner, history, outcome = game.play()
        if (winner[0] == player1 or winner[1] == player1):
            win = win + 1
    print("Use Q-Learning, after 10 iteration training, the Player1's win rate in 10000 games is: ")
    print(win / 10000)

    ex1(100)      # Training for 10000 times
    win = 0;
    for i in range(10000):
        player1 = QLearningPlayer()
        player2 = RandomPlayer()
        player3 = RandomPlayer()
        game = Board(player1, player2, player3)
        game.apply_move((0, 0))
        game.apply_move((1, 1))
        game.apply_move((2, 2))
        winner, history, outcome = game.play()
        if (winner[0] == player1 or winner[1] == player1):
            win = win + 1
    print("Use Q-Learning, after 100 iteration training, the Player1's win rate in 10000 games is: ")
    print(win / 10000)

    ex1(1000)  # Training for 10000 times
    win = 0;
    for i in range(10000):
        player1 = QLearningPlayer()
        player2 = RandomPlayer()
        player3 = RandomPlayer()
        game = Board(player1, player2, player3)
        game.apply_move((0, 0))
        game.apply_move((1, 1))
        game.apply_move((2, 2))
        winner, history, outcome = game.play()
        if (winner[0] == player1 or winner[1] == player1):
            win = win + 1
    print("Use Q-Learning, after 1000 iteration training, the Player1's win rate in 10000 games is: ")
    print(win / 10000)

