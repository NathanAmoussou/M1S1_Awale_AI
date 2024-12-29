import csv, json, os

def write_game_to_csv(csv_filename, game_data, fieldnames=None):
    """
    Write a single game's data to a CSV file.

    Parameters:
        csv_filename (str): The path to the CSV file.
        game_data (dict): A dictionary containing game data.
        fieldnames (list, optional): List of CSV column names. If None, use keys from game_data.
    """
    # Check if file exists to write header
    file_exists = os.path.isfile(csv_filename)

    with open(csv_filename, mode='a', newline='', encoding='utf-8') as csvfile:
        if fieldnames is None:
            fieldnames = list(game_data.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        # Serialize 'moves' as JSON string
        game_data_serialized = game_data.copy()
        game_data_serialized['moves'] = json.dumps(game_data_serialized['moves'])

        writer.writerow(game_data_serialized)

def write_games_to_csv(csv_filename, games_data, fieldnames=None):
    """
    Write multiple games' data to a CSV file.

    Parameters:
        csv_filename (str): The path to the CSV file.
        games_data (list): A list of dictionaries, each containing game data.
        fieldnames (list, optional): List of CSV column names. If None, use keys from first game_data.
    """
    if not games_data:
        return

    # Determine fieldnames from the first game
    if fieldnames is None:
        fieldnames = list(games_data[0].keys())

    with open(csv_filename, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header only if file does not exist
        if not os.path.isfile(csv_filename):
            writer.writeheader()

        for game_data in games_data:
            # Serialize 'moves' as JSON string
            game_data_serialized = game_data.copy()
            game_data_serialized['moves'] = json.dumps(game_data_serialized['moves'])
            writer.writerow(game_data_serialized)
