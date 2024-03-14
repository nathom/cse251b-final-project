import os
import numpy as np
import torch
import random
import re
import matplotlib.pyplot as plt

from game import *


def hist_max_val(max_val,fname):
    # plots the max_values per game -- array of ints
    total_games = len(max_val)
    # Create 'plots' directory if it doesn't exist
    if not os.path.isdir('plots/max_val'):
        os.mkdir('plots/max_val')

    max_values, count = np.unique(max_val, return_counts = True)
    n_count = count/total_games
    
    plt.bar(max_values,n_count, label='Maximum Tile Value')
    plt.xlabel('Game')
    plt.ylabel('Maximum Tile Value')
    plt.title(f'Maximum Tile Value across {total_games} games')
    plt.legend()

    plt.savefig("./plots/max_val/" + fname + ".png")
    plt.savefig("./plots/max_val/" + fname + ".svg")
    plt.close()
    
def hist_num_merges(num_merges,fname):
    # plots the number of merges per game -- array of ints
    total_games = len(num_merges)
    # Create 'plots' directory if it doesn't exist
    if not os.path.isdir('plots/num_merges'):
        os.mkdir('plots/num_merges')

    plt.bar(num_merges, label='Number of Merges')
    plt.xlabel('Game')
    plt.ylabel('Number of merges')
    plt.title(f'Number of merges across {total_games} games')
    plt.legend()

    plt.savefig("./plots/num_merges/" + fname + ".png")
    plt.savefig("./plots/num_merges/" + fname + ".svg")
    plt.close()

def hist_merge_scores(merge_scores, fname):
    # plots the merge scores per game -- merge_scores is an array scores
    total_games = len(merge_scores)
    # Create 'plots' directory if it doesn't exist
    if not os.path.isdir('plots/merge_scores'):
        os.mkdir('plots/merge_scores')

    plt.bar(merge_scores, label='Merge Scores')
    plt.xlabel('Game')
    plt.ylabel('Merge Scores')
    plt.title(f'Merge Scores across {total_games} games')
    plt.legend()

    plt.savefig("./plots/merge_scores/" + fname + ".png")
    plt.savefig("./plots/merge_scores/" + fname + ".svg")
    plt.close()

def tile_hist(tiles, fname):
    # plots the normalized dist. of tiles -- tiles is a list of lists, each list contains the tiles values for a game 
    if not os.path.isdir('plots/tiles_hits'):
        os.mkdir('plots/tiles_hits')
    
    total_games = len(tiles)
    flat_tiles = [item for sublist in tiles for item in sublist]
    
    tiles_values, count = np.unique(flat_tiles, return_counts = True)
    n_count = count/total_games
    
    plt.figure(figsize=(10, 6))
    plt.bar(tiles_values, n_count, color='skyblue')
    plt.xlabel('Tile Value')
    plt.ylabel('Normalized Distribution')
    plt.title(f'Normalized Distribution of Tile Values Over {total_games} Games')
    plt.xticks(tiles_values)
    plt.grid(axis='y')

    plt.savefig("./plots/tiles_hits/" + fname + ".png")
    plt.savefig("./plots/tiles_hits/" + fname + ".svg")
    plt.close()
