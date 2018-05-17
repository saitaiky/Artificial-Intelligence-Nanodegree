"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.



"""
import random
import math

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass

def centrality(game, move):
    """Sai: Measures the distance from the center of a certain location in the board."""
    """It is a defensive heuristic to push potenial move as close to center as possible"""
    x, y = move
    cx, cy = (math.ceil(game.width / 2), math.ceil(game.height / 2))
    return (game.width - cx) ** 2 + (game.height - cy) ** 2 - (x - cx) ** 2 - (y - cy) ** 2

def strategy_moves(game, player_legal_moves, opponent_legal_moves):
    """Sai: It is a offensive heuristic to steal a move opponent attempts to move"""
    """Which a move needs to be close enough to the central area"""
    common_moves = player_legal_moves and opponent_legal_moves
    if not common_moves:
        return 0
    # if they got more than half of the board area common moves, then agent won't penalize that move.
    enough_common_moves = math.ceil(game.height / 2) - len(common_moves)
    return max(centrality(game, m) for m in common_moves) + enough_common_moves

def euclidean_distance_centrality(game, move):

    x, y = move
    cx, cy = (math.ceil(game.width / 2), math.ceil(game.height / 2))
    return math.sqrt((x - cx)**2 + (y - cy)**2)

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_winner(player) or game.is_loser(player):
        return game.utility(player)

    # Being in a center in early game (less than 25% of board empty)
    # is good for player and bad for opponent
    game_state_factor = 2
    if len(game.get_blank_spaces()) < game.width * game.height / 4.:
        game_state_factor = 1

    opp = game.get_opponent(player)
    p_moves = game.get_legal_moves()
    opp_moves = game.get_legal_moves(opp)
    count_p_moves = len(p_moves)
    count_opp_moves = len(opp_moves)

    return float(count_p_moves - count_opp_moves + sum(centrality(game, m) for m in p_moves)*game_state_factor + strategy_moves(game, p_moves, opp_moves))


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_winner(player) or game.is_loser(player):
        return game.utility(player)

    opponent = game.get_opponent(player)
    moves_own = len(game.get_legal_moves(player))
    moves_opp = len(game.get_legal_moves(opponent))
    board = game.height * game.width
    moves_board = game.move_count / board

    # More than 33% space in the board is available(A early game)
    # True: Play agressive to gain more moves, False: Play defensive
    if moves_board > 0.33:
        move_diff = (moves_own - moves_opp*2)
    else:
        move_diff = (moves_own - moves_opp)

    # Get current move of both player and opponent
    pos_own = game.get_player_location(player)
    pos_opp = game.get_player_location(opponent)

    # abs(), return absolute value. The larger value, then longer distance
    m_distance = abs(pos_own[0] - pos_opp[0]) + abs(pos_own[1] - pos_opp[1])

    # When a move can lead to a huge move_diff and distance is near opponent
    # that is a good move.
    return float(move_diff / m_distance)


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_winner(player) or game.is_loser(player):
        return game.utility(player)

    opponent = game.get_opponent(player)
    own_moves = game.get_legal_moves()
    opp_moves = game.get_legal_moves(opponent)
    count_own_moves = len(own_moves)
    count_opp_moves = len(opp_moves)

    # Euclidean distance: measures the length of a segment connecting the two points.
    if(count_own_moves):
        own_closet_center_move = min( [euclidean_distance_centrality(game, own_move) for own_move in own_moves ] )
    else:
        own_closet_center_move = 1

    if(count_opp_moves):
        opp_awy_center_move = max( [euclidean_distance_centrality(game, opp_move) for opp_move in opp_moves ] )
    else:
        opp_awy_center_move = 1
    deffensive_factor = own_closet_center_move - opp_awy_center_move

    # More than 33% space in the board is available(A early game)
    # True: Play agressive to gain more moves, False: Play defensive
    board = game.height * game.width
    moves_board = game.move_count / board
    if moves_board > 0.33:
        offensive_factor = (count_own_moves - count_opp_moves*2)
    else:
        offensive_factor = (count_own_moves - count_opp_moves)

    return float(offensive_factor + deffensive_factor)


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

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
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        # The try/except block will automatically catch the exception
        # raised when the timer is about to expire.
        try:
            # Sai: Fixed-depth search
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        best_score = float("-inf")

        # Start from the player holding initiative in the current game state.
        # so we will call min_value first
        best_move = (-1, -1)
        for move in game.get_legal_moves():
            next_ply = game.forecast_move(move);
            score = self.min_value(next_ply, depth-1)
            # ????????? I am not sure if that is cosrrect to use larger here?????????
            if score > best_score:
                best_move = move
                best_score = score

        return best_move

    def terminal_test(self, game, depth):
        """ Return True if the game is over OR reached the depath limit
        for the active player and False otherwise.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        if len(game.get_legal_moves())==0 or depth==0:
            return True ;
        else:
            return False;

    def min_value(self, game, depth):
        """ Return the value for a win (+1) if the game is over,
        otherwise return the minimum value over all legal child
        nodes.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        if self.terminal_test(game, depth):
            return self.score(game, self);  # by Assumption 2 - 係min尼層無棋行，上一層個max嬴左。

        v = float("inf")
        for m in game.get_legal_moves():
            v = min(v, self.max_value(game.forecast_move(m),  depth-1))
        return v


    def max_value(self, game, depth):
        """ Return the value for a loss (-1) if the game is over,
        otherwise return the maximum value over all legal child
        nodes.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        if self.terminal_test(game, depth):
            return self.score(game, self);  # by assumption 2 - 係max尼層無棋行，上一層個min嬴左。

        v = float("-inf")
        for m in game.get_legal_moves():
            v = max(v, self.min_value(game.forecast_move(m), depth-1))
        return v

class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
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
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        # Sai: Iterative deepening search
        for i in range(1, 10000):

            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            try:
                best_move = self.alphabeta(game, i)
            except SearchTimeout:
                break  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        best_score = float("-inf")

        # Start from the player holding initiative in the current game state.
        # so we will call min_value first
        best_move = (-1, -1)
        for move in game.get_legal_moves():
            next_ply = game.forecast_move(move)
            score = self.min_value(next_ply, depth-1, alpha, beta)

            if score > best_score:
                best_move = move
                best_score = score
            alpha = max(alpha, best_score) # Sai: Need to update alpha too!

        return  best_move

    def terminal_test(self, game, depth):
        """ Return True if the game is over OR reached the depath limit
        for the active player and False otherwise.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        if len(game.get_legal_moves())==0 or depth==0:
            return True ;
        else:
            return False;

    def min_value(self, game, depth, alpha, beta):
        """ Return the value for a win (+1) if the game is over,
        otherwise return the minimum value over all legal child
        nodes.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        if self.terminal_test(game, depth):
            return self.score(game, self);  # by Assumption 2 - 係min尼層無棋行，上一層個max嬴左。

        v = float("inf")
        for m in game.get_legal_moves():
            v = min(v, self.max_value(game.forecast_move(m),  depth-1, alpha, beta))
            if v <= alpha:
                return v
            beta = min(beta, v) # Update in any case whether if v>=beta is true or not

        return v


    def max_value(self, game, depth, alpha, beta):
        """ Return the value for a loss (-1) if the game is over,
        otherwise return the maximum value over all legal child
        nodes.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        if self.terminal_test(game, depth):
            return self.score(game, self);  # by assumption 2 - 係max尼層無棋行，上一層個min嬴左。

        v = float("-inf")
        for m in game.get_legal_moves():
            v = max(v, self.min_value(game.forecast_move(m), depth-1, alpha, beta))
            if v >= beta:
                return v
            alpha = max(alpha, v) # Update in any case whether if v>=beta is true or not

        return v
