import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from collections import Counter

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
            losing_config = list(map(int, row[5].split(',')))
            seconds = float(row[6])

            # Store the parsed data in a dictionary
            parsed_data.append({
                'Game Number': game_number,
                'Number of Moves': num_moves,
                'Score': score,
                'Largest Tile': largest_tile,
                'Sum of Tiles': sum_of_tiles,
                'Losing Configuration': losing_config,
                'Seconds': seconds
            })
    
    # List of each entry as a dictionary
    return parsed_data


# I need to plot freq plot for highest tile for each branch cnt
# Plot branching factor vs average score

def hist_freq_tile(name, value, largest_tiles):
    if not os.path.isdir("./data"): os.mkdir("./data")

    # Get the counts of each unique value
    a = Counter(largest_tiles).most_common()
    a.sort(key=lambda x: x[0])

    # Prepare the values for plotting
    num, freq = [], []
    for (v, c) in a:
        num.append(str(v))
        freq.append(c)

    print(num, freq)
    plt.bar(num, freq)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Branch ' + str(value) + ' Highest Tile Frequency Histogram')
    
    plt.savefig("./data/" + name + ".png")
    plt.cla()

def avg_branch(name, branches, scores):
    if not os.path.isdir("./data"): os.mkdir("./data")

    plt.plot(branches, scores)
    plt.title('Branch Factor vs Average Score')

    plt.savefig("./data/" + name + ".png")
    plt.cla()

csv_list = [("./data/monte_carlo_branch=100_ngames=20_method=1.csv", "branches_100_games_20", 100),
            ("./data/monte_carlo_branch=200_ngames=20_method=1.csv", "branches_200_games_20", 200),
            ("./data/monte_carlo_branch=300_ngames=20_method=1.csv", "branches_300_games_20", 300),
            ("./data/monte_carlo_branch=400_ngames=20_method=1.csv", "branches_400_games_20", 400),
            ("./data/monte_carlo_branch=500_ngames=20_method=1.csv", "branches_500_games_20", 500)]

# Kinda hard coded, but it can be streamlined with a function declaration
scores = []
branches = [100, 200, 300, 400, 500]
for (f, n, v) in csv_list:
    data = csvReader(f)

    largest_tile = []
    score = 0
    for d in data:
        largest_tile.append(d['Largest Tile'])
        score += d['Score']

    score /= len(data)
    scores.append(score)

    hist_freq_tile(n, v, largest_tile)
avg_branch("avg_score_branch_plot", branches, scores)
