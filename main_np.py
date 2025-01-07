#main_np.py
from board_rules_interface_np import AwaleGame
import agents_np
import data_export_np
import os
import re
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("game_generation.log"),
        logging.StreamHandler()
    ]
)

def get_next_filename(directory, agent1_class, agent2_class, num_games):
    """
    Generate the next available filename with an incremented numerical prefix.

    Parameters:
        directory (str): Directory where CSV files are stored.
        agent1_class (class): Class of the first agent.
        agent2_class (class): Class of the second agent.
        num_games (int): Number of games run in each file.

    Returns:
        str: The next available filename.
    """
    # Define the filename pattern with numerical prefix
    pattern = rf'^(\d+)-{agent1_class.__name__}-vs-{agent2_class.__name__}-{num_games}\.csv$'
    regex = re.compile(pattern)

    # Initialize the highest number found
    highest_num = -1

    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Iterate over files in the directory to find the highest numerical prefix
    for filename in os.listdir(directory):
        match = regex.match(filename)
        if match:
            num = int(match.group(1))
            if num > highest_num:
                highest_num = num

    # Determine the next number
    next_num = highest_num + 1

    # Format the numerical prefix with leading zeros (e.g., 00, 01, ..., 99)
    if next_num < 10:
        num_str = f'0{next_num}'
    else:
        num_str = f'{next_num}'

    # Construct the new filename
    new_filename = f'{num_str}-{agent1_class.__name__}-vs-{agent2_class.__name__}-{num_games}.csv'
    full_path = os.path.join(directory, new_filename)
    return full_path

def run_multiple_games(num_games, agent1_class, agent2_class, max_time=2, directory='game_datas'):
    """
    Run multiple Awale games between two agents and log the results to a CSV file.

    Parameters:
        num_games (int): Number of games to run.
        agent1_class (class): Class of the first agent.
        agent2_class (class): Class of the second agent.
        max_time (float): Maximum computation time per move in seconds.
        csv_filename (str): Name of the CSV file to write results to.
    """

    csv_filename = get_next_filename(directory, agent1_class, agent2_class, num_games)

    logging.info(f"Exporting games to: {csv_filename}")

    # Record the start time
    start_time = time.time()
    logging.info("Simulation started.")

    for game_id in range(1, num_games + 1):
        # Instantiate agents
        player1_agent = agent1_class() if (agent1_class.__name__ == 'RandomAgent' or agent1_class.__name__ == 'HumanAgent') else agent1_class(max_time=max_time)
        player2_agent = agent2_class() if (agent2_class.__name__ == 'RandomAgent' or agent2_class.__name__ == 'HumanAgent') else agent2_class(max_time=max_time)

        # Initialize the game with the selected agents and assign game_id
        game = AwaleGame(player1_agent=player1_agent, player2_agent=player2_agent, game_id=game_id)
        game.run_game()

        # Retrieve game data
        game_data = game.get_game_data()

        # Use data_export module to write to CSV
        data_export_np.write_game_to_csv(csv_filename, game_data)

        logging.info(f"Game {game_id}/{num_games} completed. Winner: {game_data['winner']}")

    # Record the end time
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Convert elapsed time to hours, minutes, seconds
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    logging.info(f"Simulation completed in {int(hours)}h {int(minutes)}m {int(seconds)}s.")


if __name__ == "__main__":
    num_games = 1
    agent1_class = agents_np.MinimaxAgent6_4
    agent2_class = agents_np.HumanAgent
    directory = 'game_datas'
    csv_filename = get_next_filename(directory, agent1_class, agent2_class, num_games)
    run_multiple_games(
        num_games=num_games,  # Adjust the number as needed
        agent1_class=agent1_class,
        agent2_class=agent2_class,
        directory=directory,
        max_time=2
    )
