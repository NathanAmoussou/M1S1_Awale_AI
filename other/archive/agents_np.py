import random
import math
import time
from functools import lru_cache
import numba_type_shit
import numpy as np

class Agent:
    """
    Abstract base class for all agents.
    """
    def get_move(self, game_state):
        """
        Determine the next move.
        Must be overridden by subclasses.

        Parameters:
            game_state (AwaleGame): The current state of the game.

        Returns:
            tuple: A tuple ((hole, color), elapsed time, depth) or (hole, color) or None.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

class HumanAgent(Agent):
    def get_move(self, game_state):
        """
        Prompt the human player to input a move.

        Parameters:
            game_state (AwaleGame): The current state of the game.

        Returns:
            tuple: A tuple (hole, color) representing the move.
        """
        while True:
            try:
                hole = int(input("Choisissez un trou (1-16) : ")) - 1
                color = int(input("Choisissez une couleur (0 = Rouge, 1 = Bleu) : "))
                if game_state.is_valid_move(hole, color):
                    return (hole, color), None, None
                else:
                    print("Coup invalide. Veuillez réessayer.")
            except ValueError:
                print("Entrée invalide. Veuillez entrer des nombres valides.")

class RandomAgent(Agent):
    def get_move(self, game_state):
        """
        Select a random valid move.

        Parameters:
            game_state (AwaleGame): The current state of the game.

        Returns:
            tuple: A tuple (hole, color) representing the move, or None if no moves are available.
        """
        valid_moves = game_state.get_valid_moves()
        if not valid_moves:
            print("Aucun coup valide disponible.")
            return None, None, None
        move = random.choice(valid_moves)
        print(f"IA aléatoire a choisi le coup: Hole {move[0]+1} Color {'R' if move[1]==0 else 'B'}")
        return move, None, None

class GPTMinimaxAgentV2(Agent):
    def __init__(self, max_time=2):
        """
        Initialize the Minimax agent.

        Parameters:
            max_time (float): Maximum time allowed for move computation in seconds.
        """
        self.max_time = max_time

    def get_move(self, game_state):
        """
        Determine the best move using the Minimax algorithm with alpha-beta pruning.

        Parameters:
            game_state (AwaleGame): The current state of the game.

        Returns:
            tuple: A tuple (hole, color) representing the best move found.
        """
        start_time = time.time()
        depth = 1
        best_move_found = None

        while True:
            elapsed_time = time.time() - start_time
            if elapsed_time >= self.max_time:
                break

            try:
                eval_val, move = self.minimax(
                    game_state.clone(),
                    depth,
                    -math.inf,
                    math.inf,
                    True,  # Maximizing player
                    start_time,
                    self.max_time
                )
                if move is not None:
                    best_move_found = move
            except Exception as e:
                print(f"Exception during minimax at depth {depth}: {e}")
                break

            depth += 1

        total_time = time.time() - start_time
        # print(f"Temps de calcul (Minimax) : {total_time:.2f}s, profondeur atteinte : {depth - 1}")
        return best_move_found, total_time, depth - 1

    def minimax(self, game_state, depth, alpha, beta, maximizing_player, start_time, max_time):
        """
        Recursive Minimax function with alpha-beta pruning.

        Parameters:
            game_state (AwaleGame): The current state of the game.
            depth (int): Current depth in the game tree.
            alpha (float): Alpha value for pruning.
            beta (float): Beta value for pruning.
            maximizing_player (bool): True if the current layer is maximizing, False otherwise.
            start_time (float): The start time of the computation.
            max_time (float): Maximum allowed computation time.

        Returns:
            tuple: (evaluation value, best move)
        """
        # Time check
        if time.time() - start_time >= max_time:
            return game_state.GPT_evaluate_V2(), None

        # Terminal condition
        if game_state.game_over() or depth == 0:
            return game_state.GPT_evaluate_V2(), None

        moves = game_state.get_valid_moves()
        if not moves:
            return game_state.GPT_evaluate_V2(), None

        best_move = None

        if maximizing_player:
            max_eval = -math.inf
            for move in moves:
                # Time check within loop
                if time.time() - start_time >= max_time:
                    break

                clone_state = game_state.clone()
                clone_state.play_move(*move)
                eval_val, _ = self.minimax(
                    clone_state,
                    depth - 1,
                    alpha,
                    beta,
                    False,  # Switch to minimizing
                    start_time,
                    max_time
                )
                if eval_val > max_eval:
                    max_eval = eval_val
                    best_move = move
                alpha = max(alpha, eval_val)
                if beta <= alpha:
                    break  # Beta cut-off
            return max_eval, best_move
        else:
            min_eval = math.inf
            for move in moves:
                if time.time() - start_time >= max_time:
                    break

                clone_state = game_state.clone()
                clone_state.play_move(*move)
                eval_val, _ = self.minimax(
                    clone_state,
                    depth - 1,
                    alpha,
                    beta,
                    True,  # Switch to maximizing
                    start_time,
                    max_time
                )
                if eval_val < min_eval:
                    min_eval = eval_val
                    best_move = move
                beta = min(beta, eval_val)
                if beta <= alpha:
                    break  # Alpha cut-off
            return min_eval, best_move

class ClaudeMinimaxAgentV1(Agent):
    def __init__(self, max_time=2):
        """
        Initialize the Minimax agent.

        Parameters:
            max_time (float): Maximum time allowed for move computation in seconds.
        """
        self.max_time = max_time
        self.nodes_cut = 0
        # flag can be: 'EXACT', 'LOWERBOUND', or 'UPPERBOUND'
        self.transposition_table = {}
        # Move ordering table stores historical success of moves
        self.move_ordering = {}

        # Constants for transposition table
        self.EXACT = 0
        self.LOWERBOUND = 1
        self.UPPERBOUND = 2

        # Maximum size for transposition table (adjust based on available memory)
        self.MAX_TABLE_SIZE = 1000000

    # claude get move
    def get_move(self, game_state):
        start_time = time.time()
        depth = 1
        best_move_found = None

        while True:
            elapsed_time = time.time() - start_time
            if elapsed_time >= self.max_time:
                break

            try:
                eval_val, move = numba_type_shit.minimax_numba(
                    game_state.board,
                    game_state.scores,
                    game_state.current_player,
                    game_state.player_holes,
                    depth,
                    -math.inf,
                    math.inf,
                    True
                )
                if move != -1:
                    best_move_found = tuple(move)
            except Exception as e:
                print(f"Exception during minimax at depth {depth}: {e}")
                break

            depth += 1

        total_time = time.time() - start_time
        # print(f"Temps de calcul (Minimax) : {total_time:.2f}s, profondeur atteinte : {depth - 1}")
        return best_move_found, total_time, depth - 1


    # claude minimax
    def minimax(self, game_state, depth, alpha, beta, maximizing_player, start_time, max_time, is_root=False):
        # Time check
        if time.time() - start_time >= max_time * 0.95:
            raise TimeoutError()

        state_hash = self._get_state_hash(game_state)
        if not is_root and state_hash in self.transposition_table:
            stored_depth, stored_value, stored_move = self.transposition_table[state_hash]
            if stored_depth >= depth:
                return stored_value, stored_move

        # Conditions terminales
        if game_state.game_over() or depth == 0:
            eval_val = game_state.claude_evaluate_V1()
            return eval_val, None

        moves = self._order_moves(game_state.get_valid_moves())
        if not moves:
            eval_val = game_state.claude_evaluate_V1()
            return eval_val, None

        best_move = None

        if maximizing_player:
            max_eval = -math.inf
            for move in moves:
                # Vérification du temps
                if time.time() - start_time >= max_time * 0.95:
                    raise TimeoutError()

                clone_state = game_state.clone()
                clone_state.play_move(*move)
                eval_val, _ = self.minimax(
                    clone_state,
                    depth - 1,
                    alpha,
                    beta,
                    False,
                    start_time,
                    max_time
                )
                if eval_val > max_eval:
                    max_eval = eval_val
                    best_move = move
                alpha = max(alpha, eval_val)
                if beta <= alpha:
                    self.nodes_cut += 1
                    break
            # Stocker dans la table de transposition
            self.transposition_table[state_hash] = (depth, max_eval, best_move)
            return max_eval, best_move
        else:
            min_eval = math.inf
            for move in moves:
                if time.time() - start_time >= max_time * 0.95:
                    raise TimeoutError()

                clone_state = game_state.clone()
                clone_state.play_move(*move)
                eval_val, _ = self.minimax(
                    clone_state,
                    depth - 1,
                    alpha,
                    beta,
                    True,
                    start_time,
                    max_time
                )
                if eval_val < min_eval:
                    min_eval = eval_val
                    best_move = move
                beta = min(beta, eval_val)
                if beta <= alpha:
                    self.nodes_cut += 1
                    break
            # Stocker dans la table de transposition
            self.transposition_table[state_hash] = (depth, min_eval, best_move)
            return min_eval, best_move

    @lru_cache(maxsize=1000000)
    def _get_state_hash(self, game_state):
        """Create a hash of the current game state."""
        return (tuple(map(tuple, game_state.board)), tuple(game_state.scores), game_state.current_player)


    def _order_moves(self, moves):
        """Order moves based on previously successful moves."""
        return sorted(moves, key=lambda m: self.move_ordering.get(m, 0), reverse=True)
