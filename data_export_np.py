#data_export.py
import csv
import json
import os
import numpy as np
from typing import Dict, List, Optional, Any

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

def serialize_numpy_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert NumPy arrays and types in game data to Python native types.

    Parameters:
        data (dict): Dictionary containing game data with potential NumPy types

    Returns:
        dict: Dictionary with all NumPy types converted to Python native types
    """
    serialized_data = {}

    for key, value in data.items():
        if isinstance(value, dict):
            serialized_data[key] = serialize_numpy_data(value)
        elif isinstance(value, list):
            serialized_data[key] = [
                serialize_numpy_data(item) if isinstance(item, dict)
                else item.tolist() if isinstance(item, np.ndarray)
                else item
                for item in value
            ]
        elif isinstance(value, np.ndarray):
            serialized_data[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating, np.bool_)):
            serialized_data[key] = value.item()
        else:
            serialized_data[key] = value

    return serialized_data

def write_game_to_csv(csv_filename: str, game_data: Dict[str, Any], fieldnames: Optional[List[str]] = None) -> None:
    """
    Write a single game's data to a CSV file, handling NumPy data types.

    Parameters:
        csv_filename (str): The path to the CSV file
        game_data (dict): A dictionary containing game data with potential NumPy types
        fieldnames (list, optional): List of CSV column names. If None, use keys from game_data
    """
    # Check if file exists to write header
    file_exists = os.path.isfile(csv_filename)

    # Ensure directory exists
    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)

    with open(csv_filename, mode='a', newline='', encoding='utf-8') as csvfile:
        if fieldnames is None:
            fieldnames = list(game_data.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        # Convert NumPy types and serialize moves
        game_data_serialized = serialize_numpy_data(game_data)
        game_data_serialized['moves'] = json.dumps(
            game_data_serialized['moves'],
            cls=NumpyEncoder
        )

        writer.writerow(game_data_serialized)

def write_games_to_csv(csv_filename: str, games_data: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    """
    Write multiple games' data to a CSV file, handling NumPy data types.

    Parameters:
        csv_filename (str): The path to the CSV file
        games_data (list): A list of dictionaries, each containing game data with potential NumPy types
        fieldnames (list, optional): List of CSV column names. If None, use keys from first game_data
    """
    if not games_data:
        return

    # Ensure directory exists
    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)

    # Determine fieldnames from the first game
    if fieldnames is None:
        fieldnames = list(games_data[0].keys())

    with open(csv_filename, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header only if file does not exist
        if not os.path.isfile(csv_filename):
            writer.writeheader()

        for game_data in games_data:
            # Convert NumPy types and serialize moves
            game_data_serialized = serialize_numpy_data(game_data)
            game_data_serialized['moves'] = json.dumps(
                game_data_serialized['moves'],
                cls=NumpyEncoder
            )
            writer.writerow(game_data_serialized)
