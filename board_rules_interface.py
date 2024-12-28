from agents import Agent

class AwaleGame:
    def __init__(self, player1_agent: Agent, player2_agent: Agent):
        self.board = [[2, 2] for _ in range(16)]
        self.scores = [0, 0]
        self.player_holes = {
            1: [i for i in range(0, 16, 2)],
            2: [i for i in range(1, 16, 2)]
        }
        self.current_player = 1
        self.player_agents = {
            1: player1_agent,
            2: player2_agent
        }
        self.turn_number = 0

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

    def GPT_evaluate_V2(self) -> int:
        """
        A more advanced evaluation function for your new Awale variant.
        Considers:
            1) Captured seeds difference
            2) Board control (seeds in my holes - seeds in opp holes)
            3) Imminent capture threats
        """
        my_index = self.current_player - 1
        opp_index = 1 - my_index

        # 1) Captured difference
        captured_diff = self.scores[my_index] - self.scores[opp_index]

        # 2) Board control
        my_holes = self.player_holes[self.current_player]
        opp_holes = self.player_holes[3 - self.current_player]
        my_seeds = sum(sum(self.board[h]) for h in my_holes)
        opp_seeds = sum(sum(self.board[h]) for h in opp_holes)
        board_control = my_seeds - opp_seeds

        # 3) Threat: number of opponent holes with 1 or 2 or 3 seeds minus my holes with 1 or 2 seeds or 3 seeds
        opp_threat = sum(1 for h in opp_holes if 1 <= sum(self.board[h]) <= 3)
        my_threat = sum(1 for h in my_holes if 1 <= sum(self.board[h]) <= 3)
        threat_score = opp_threat - my_threat

        # Weighted sum
        # Example weights: 50 for captured seeds (heavily important),
        #                  5 for board control,
        #                  3 for threat potential
        # EVAL = 50 * captured_diff + 5 * board_control + 3 * threat_score
        EVAL = 3 * captured_diff + 5 * board_control + 50 * threat_score
        return EVAL

    # Claude AI evaluation function
    def claude_evaluate_V1(self) -> int:
        """
        Enhanced evaluation function with multiple strategic considerations.
        """
        my_index = self.current_player - 1
        opp_index = 1 - my_index

        # 1. Score difference (most important)
        score_diff = self.scores[my_index] - self.scores[opp_index]

        # 2. Control of the board
        my_holes = self.player_holes[self.current_player]
        opp_holes = self.player_holes[3 - self.current_player]

        # Calculate seeds under control
        my_seeds = {
            'red': sum(self.board[h][0] for h in my_holes),
            'blue': sum(self.board[h][1] for h in my_holes)
        }
        opp_seeds = {
            'red': sum(self.board[h][0] for h in opp_holes),
            'blue': sum(self.board[h][1] for h in opp_holes)
        }

        # 3. Capture opportunities
        capture_potential = 0
        for hole in range(16):
            total_seeds = sum(self.board[hole])
            if total_seeds in [1, 4]:  # One move away from capture
                if hole in my_holes:
                    capture_potential += 2
                else:
                    capture_potential -= 2

        # 4. Mobility score (number of possible moves)
        my_mobility = sum(1 for h in my_holes for c in [0, 1] if self.board[h][c] > 0)
        opp_mobility = sum(1 for h in opp_holes for c in [0, 1] if self.board[h][c] > 0)
        mobility_score = my_mobility - opp_mobility

        # 5. Seed distribution (prefer spread out seeds)
        my_distribution = sum(1 for h in my_holes if sum(self.board[h]) > 0)
        opp_distribution = sum(1 for h in opp_holes if sum(self.board[h]) > 0)
        distribution_score = my_distribution - opp_distribution

        # 6. End game considerations
        total_seeds = sum(sum(hole) for hole in self.board)
        if total_seeds < 16:  # End game is near
            score_diff_weight = 100  # Increase importance of actual score
        else:
            score_diff_weight = 50

        # Weighted sum of all factors
        return (
            score_diff_weight * score_diff +
            30 * (sum(my_seeds.values()) - sum(opp_seeds.values())) +
            20 * capture_potential +
            15 * mobility_score +
            10 * distribution_score
        )

    def is_valid_move(self, hole, color):
        if hole is None or color is None:
            return False
        if hole not in self.player_holes[self.current_player]:
            return False
        if color not in [0, 1]:
            return False
        if self.board[hole][color] == 0:
            return False
        return True

    def get_valid_moves(self):
        moves = []
        for hole in self.player_holes[self.current_player]:
            for color in [0, 1]:
                if self.board[hole][color] > 0:
                    moves.append((hole, color))
        return moves

    def play_move(self, hole, color):
        if not self.is_valid_move(hole, color):
            raise ValueError("Mouvement invalide !")

        seeds_to_sow = self.board[hole][color]
        self.board[hole][color] = 0

        initial_hole = hole
        current_index = hole

        while seeds_to_sow > 0:
            current_index = (current_index + 1) % 16

            if current_index == initial_hole:
                continue

            if color == 0:
                if current_index in self.player_holes[self.current_player]:
                    continue
                self.board[current_index][color] += 1
                seeds_to_sow -= 1
            else:
                self.board[current_index][color] += 1
                seeds_to_sow -= 1

        self.apply_capture(current_index)
        self.current_player = 3 - self.current_player

    def apply_capture(self, start_hole):
        current_index = start_hole
        while True:
            total_seeds = sum(self.board[current_index])
            if total_seeds in [2, 3]:
                self.scores[self.current_player - 1] += total_seeds
                self.board[current_index] = [0, 0]
                current_index = (current_index - 1) % 16
            else:
                break

    def clone(self):
        """
        Create a deep copy of the game state.
        """
        cloned_game = AwaleGame(
            player1_agent=self.player_agents[1],
            player2_agent=self.player_agents[2]
        )
        cloned_game.board = [hole.copy() for hole in self.board]
        cloned_game.scores = self.scores.copy()
        cloned_game.current_player = self.current_player
        return cloned_game

    def game_over(self):
        total_seeds = sum(sum(hole) for hole in self.board)
        if total_seeds < 8:
            return True
        if self.scores[0] >= 33 or self.scores[1] >= 33:
            return True
        if self.scores[0] == 32 and self.scores[1] == 32:
            return True
        if self.turn_number >= 150:
            return True
        return False

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
            if move == (None, None, None):
                break
            (hole, color), compute_time, depth_reached = move
            try:
                self.play_move(hole, color)
            except ValueError as e:
                print(e)
                self.turn_number -= 1
                continue

            self.display_board(
                turn_number=self.turn_number,
                last_move=(self.current_player, (hole, color)),
                depth_reached=depth_reached,
                calc_time=compute_time
            )

        self.display_game_end(self.player_agents)
