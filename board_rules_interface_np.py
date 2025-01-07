#board_rules_interface_np.py
import numpy as np
from typing import List, Tuple

class AwaleGame:
    def __init__(self, player1_agent, player2_agent, game_id=None):
        # Use NumPy arrays instead of nested lists
        self.board = np.zeros((16, 2), dtype=np.int8)  # dtype=int8 for memory efficiency
        self.board.fill(2)  # Initialize with 2 seeds each
        self.scores = np.zeros(2, dtype=np.int16)
        # Pre-compute player holes as NumPy arrays
        self.player_holes = {
            1: np.arange(0, 16, 2, dtype=np.int8),
            2: np.arange(1, 16, 2, dtype=np.int8)
        }
        self.current_player = 1
        self.player_agents = {
            1: player1_agent,
            2: player2_agent
        }
        self.turn_number = 0
        self.game_id = game_id
        self.moves_log = []

    def display_board(self, turn_number=0, last_move=None, depth_reached=None, calc_time=None):
        # Display last move if provided
        if last_move:
            player_num, move = last_move
            formated_last_move = ''
            if last_move[1][0] < 9:
                formated_last_move = f'0{last_move[1][0] + 1}'
            else:
                formated_last_move = f'{last_move[1][0] + 1}'
            if last_move[1][1] == 0:
                formated_last_move += 'R'
            else:
                formated_last_move += 'B'
            if depth_reached is not None and calc_time is not None:
                print(f"\nJ{3 - int(player_num)} ({self.player_agents[((turn_number - 1) % 2) + 1].__class__.__name__}[{calc_time:.2f}s, {depth_reached} depth]): {formated_last_move}")
            else:
                print(f"\nJ{3 - int(player_num)} ({self.player_agents[((turn_number - 1) % 2) + 1].__class__.__name__}): {formated_last_move}")

        # Display turn and scores header
        scores_str = f"T{turn_number if turn_number is not None else '?'} (J1={self.scores[0]}, J2={self.scores[1]}):\n"
        print(f"\n{scores_str}")

        # Display hole numbers
        holes_str = "  N: " + " ".join(f"{i+1:02d}" for i in range(16))
        print(holes_str)

        # Display separator
        separator = "     " + "-" * 47
        print(separator)

        # Display red seeds
        red_seeds = [f"{hole[0]:02d}" for hole in self.board]
        print("  R: " + " ".join(red_seeds))

        # Display blue seeds
        blue_seeds = [f"{hole[1]:02d}" for hole in self.board]
        print("  B: " + " ".join(blue_seeds))

        # Display separator
        print(separator)

        # Display total seeds in each hole
        total_seeds = [f"{sum(hole):02d}" for hole in self.board]
        print("  T: " + " ".join(total_seeds))

    def display_game_end(self, player_types):
        print("\nENDGAME\n")

        if self.scores[0] == self.scores[1]:
            print("WINNER: TIE")
        elif self.turn_number >= 150:
            print("WINNER: TIE (turn limit reached)")
        else:
            winner = 1 if self.scores[0] > self.scores[1] else 2
            print(f"WINNER: J{winner} ({player_types[winner].__class__.__name__})")

        print("\nSCORE:")
        print(f"  J1 ({player_types[1].__class__.__name__}): {self.scores[0]}")
        print(f"  J2 ({player_types[2].__class__.__name__}): {self.scores[1]}")
        print()

    def is_valid_move(self, hole: int, color: int) -> bool:
        if hole is None or color is None:
            return False
        return (hole in self.player_holes[self.current_player] and
                0 <= color <= 1 and
                self.board[hole, color] > 0)

    def get_valid_moves(self) -> List[Tuple[int, int]]:
        # Vectorized valid moves computation
        player_holes = self.player_holes[self.current_player]
        moves = []
        # Use NumPy boolean indexing
        valid_red = player_holes[self.board[player_holes, 0] > 0]
        valid_blue = player_holes[self.board[player_holes, 1] > 0]

        moves.extend((hole, 0) for hole in valid_red)
        moves.extend((hole, 1) for hole in valid_blue)
        return moves

    def play_move(self, hole: int, color: int) -> None:
        if not self.is_valid_move(hole, color):
            raise ValueError("Invalid move!")

        seeds_to_sow = self.board[hole, color]
        self.board[hole, color] = 0

        # Create a mask for opponent's holes
        opponent_holes = np.ones(16, dtype=bool)
        opponent_holes[self.player_holes[self.current_player]] = False

        current_index = hole
        while seeds_to_sow > 0:
            current_index = (current_index + 1) % 16
            if current_index == hole or self.board[current_index].sum() >= 16:
                continue

            if color == 0:  # Red seeds
                if opponent_holes[current_index]:
                    self.board[current_index, color] += 1
                    seeds_to_sow -= 1
            else:  # Blue seeds
                self.board[current_index, color] += 1
                seeds_to_sow -= 1

        self.apply_capture(current_index)
        self.current_player = 3 - self.current_player

    def apply_capture(self, start_hole: int) -> None:
        current_index = start_hole
        while True:
            total_seeds = np.sum(self.board[current_index])
            if total_seeds in [2, 3]:
                self.scores[self.current_player - 1] += total_seeds
                self.board[current_index] = 0
                current_index = (current_index - 1) % 16
            else:
                break

    def clone(self):
        cloned_game = AwaleGame(
            player1_agent=self.player_agents[1],
            player2_agent=self.player_agents[2]
        )
        # Use NumPy's copy method
        cloned_game.board = self.board.copy()
        cloned_game.scores = self.scores.copy()
        cloned_game.current_player = self.current_player
        return cloned_game

    def game_over(self) -> bool:
        total_seeds = np.sum(self.board)
        return (total_seeds < 8 or
                np.max(self.scores) >= 33 or
                (self.scores[0] == 32 and self.scores[1] == 32) or
                self.turn_number >= 150)

    def get_winner(self):
        if self.scores[0] > self.scores[1]:
            return "Joueur 1"
        elif self.scores[1] > self.scores[0]:
            return "Joueur 2"
        else:
            return "Égalité"

    def get_move_for_current_player(self):
        current_agent = self.player_agents[self.current_player]
        move = current_agent.get_move(self)
        return move

    def run_game(self):
        self.display_board(turn_number=self.turn_number)

        while not self.game_over():
            self.turn_number += 1
            move = self.get_move_for_current_player()
            if move == (None, None, None) or move[0] == None:
                break
            (hole, color), compute_time, depth_reached = move

            board_before_move = [hole.copy() for hole in self.board]

            try:
                self.play_move(hole, color)
            except ValueError as e:
                print(e)
                self.turn_number -= 1
                continue

            self.moves_log.append({
                'turn_number': self.turn_number,
                'player': self.current_player,
                'board_before_move': board_before_move,
                'move': {'hole': hole, 'color': color},
                'compute_time': compute_time,
                'depth_reached': depth_reached
            })

            self.display_board(
                turn_number=self.turn_number,
                last_move=(self.current_player, (hole, color)),
                depth_reached=depth_reached,
                calc_time=compute_time
            )

        self.display_game_end(self.player_agents)

    def get_game_data(self):
        """
        Retrieve all relevant game data for logging.
        """
        return {
            'game_id': self.game_id,
            'player1_agent': self.player_agents[1].__class__.__name__,
            'player2_agent': self.player_agents[2].__class__.__name__,
            'moves': self.moves_log,
            'winner': self.get_winner(),
            'player1_score': self.scores[0],
            'player2_score': self.scores[1],
            'number_of_turns': self.turn_number
        }
