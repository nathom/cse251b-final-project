##
## PURPOSE: FOR MCTS CSV FILE
##
import sys
import csv

def parse_csv(filename):
    # Initialize variables to store cumulative values
    total_moves = 0
    total_score = 0
    total_largest_tile = 0
    total_sum_of_tiles = 0
    total_seconds = 0
    num_games = 0

    # Read the CSV file
    with open(filename, 'r') as file:
        reader = csv.DictReader(file, delimiter=',')
        next(reader, None)
        for row in reader:
            num_games += 1
            total_moves += int(row['Number of Moves'])
            total_score += int(row['Score'])
            total_largest_tile += int(row['Largest Tile'])
            total_sum_of_tiles += int(row['Sum of Tiles'])
            total_seconds += float(row['seconds'])

    # Calculate averages
    avg_moves = total_moves / num_games
    avg_score = total_score / num_games
    avg_largest_tile = total_largest_tile / num_games
    avg_sum_of_tiles = total_sum_of_tiles / num_games
    avg_seconds = total_seconds / num_games

    # Print the averages
    print("Average number of moves:", avg_moves)
    print("Average score:", avg_score)
    print("Average largest tile:", avg_largest_tile)
    print("Average sum of tiles:", avg_sum_of_tiles)
    print("Average seconds:", avg_seconds)

parse_csv(sys.argv[1])
