import cProfile
import pstats
import io
from agents_np_nb import ClaudeMinimaxAgentV1, RandomAgent
from board_rules_interface_np import AwaleGame

def profile_minimax(agent, opponent_agent, num_games=1):
    """
    Profile the Minimax agent by running it against an opponent agent.

    Parameters:
        agent (Agent): The Minimax agent to profile.
        opponent_agent (Agent): The opponent agent.
        num_games (int): Number of games to run for profiling.
    """
    profiler = cProfile.Profile()
    profiler.enable()

    for game_number in range(1, num_games + 1):
        print(f"\n=== Profiling Game {game_number} ===")
        game = AwaleGame(player1_agent=agent, player2_agent=opponent_agent, game_id=game_number)
        game.run_game()

    profiler.disable()

    # Capture the profiling results
    s = io.StringIO()
    sortby = 'cumtime'
    ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
    ps.print_stats(20)  # Affiche les 20 fonctions les plus consommatrices

    print("\n=== Profiling Results ===")
    print(s.getvalue())

if __name__ == "__main__":
    # Initialiser les agents
    minimax_agent = ClaudeMinimaxAgentV1(max_time=2)
    random_agent = RandomAgent()

    # Profilage : Minimax vs RandomAgent
    profile_minimax(minimax_agent, random_agent, num_games=3)
