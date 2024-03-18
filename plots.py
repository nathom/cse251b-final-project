import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from collections import Counter

from metrics import *

def csvReader(csv_path):
    parsed_data = []

    # Open the CSV file and read its contents
    with open(csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)

        for row in reader:
            # Convert elements to appropriate data types
            game_number = int(row[0])
            num_moves = int(row[1])
            score = int(row[2])
            largest_tile = int(row[3])
            sum_of_tiles = int(row[4])
            num_merges = int(row[5])
            losing_config = list(map(int, row[6].split(',')))
            seconds = float(row[7])

            # Store the parsed data in a dictionary
            parsed_data.append({
                'Game Number': game_number,
                'Number of Moves': num_moves,
                'Score': score,
                'Largest Tile': largest_tile,
                'Sum of Tiles': sum_of_tiles,
                'Number of Merges': num_merges,
                'Losing Configuration': losing_config,
                'Seconds': seconds
            })
    
    # List of each entry as a dictionary
    return parsed_data

csv_list = [("./data/monte_carlo_branch=200_ngames=5_method=2.csv", "branches_200_games_5", 200),
            ("./data/monte_carlo_branch=5_ngames=50_method=2.csv", "branches_5_games_50", 5),
            # ("./data/monte_carlo_branch=300_ngames=20_method=1.csv", "branches_300_games_20", 300),
            # ("./data/monte_carlo_branch=400_ngames=20_method=1.csv", "branches_400_games_20", 400),
            ("./data/monte_carlo_branch=10_ngames=1_method=2.csv", "branches_10_games_1", 10)]

# Kinda hard coded, but it can be streamlined with a function declaration
scores = []
max_val = []
branches = [100, 5, 10]
for (f, n, v) in csv_list:
    data = csvReader(f)
    tiles = []
    largest_tile = []
    num_merges = []
    score = 0
    for d in data:
        largest_tile.append(d['Largest Tile'])
        score += d['Score']
        tiles.append(d['Losing Configuration'])
        num_merges.append(d['Number of Merges'])
    
    score /= len(data)
    scores.append(score)
    print (largest_tile)
    hist_tiles(tiles, n, "mcts with " + str(v)+ " branches") # tiles distribution 
    hist_max_val(largest_tile, n, "mcts with " + str(v)+ " branches") # largest tile distribution
    hist_num_merges(num_merges, n, "mcts with " + str(v)+ " branches") # num merges distribution
print (scores)
avg_branch("avg_score_branch_plot", branches, scores)
