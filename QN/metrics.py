import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def hist_max_val(max_val,fname, title_suf = ""):
    print ("plotting max_val")
    # plots the max_values per game -- array of ints
    total_games = len(max_val)
    # Create 'plots' directory if it doesn't exist
    if not os.path.isdir('plots/max_val'):
        os.mkdir('plots/max_val')

    max_values, count = np.unique(max_val, return_counts = True)
    n_count = count/total_games
    
    max_values = [str(x) for x in max_values]
    print (max_values, n_count)

    plt.figure(figsize=(10, 6))

    plt.bar(max_values,n_count, label='Maximum Tile Value', color='skyblue')
    plt.xlabel('Maximum Tile Value')
    plt.ylabel('Normalized Frequency')
    if title_suf == "":
        plt.title(f'Maximum Tile Value across {total_games} games')
    else:
        plt.title(f'Maximum Tile Value across {total_games} games - {title_suf}')
    plt.legend()
    plt.grid(axis='y')

    plt.savefig("./plots/max_val/" + fname + ".png")
    plt.savefig("./plots/max_val/" + fname + ".svg")
    plt.cla()
    
def hist_num_merges(num_merges,fname, title_suf = "", bs = 50):
    print ("plotting num_merges")
    # plots the number of merges per game -- array of ints
    total_games = len(num_merges)
    # Create 'plots' directory if it doesn't exist
    if not os.path.isdir('plots/num_merges'):
        os.mkdir('plots/num_merges')
    
    num_merges, count = np.unique(num_merges, return_counts = True)
    # put the values in buckets of 100
    #print (num_merges, count)
    # print ("Putting values in buckets of 50")
    min_merges = min(num_merges) 
    min_merges = min_merges - (min_merges % bs)
    max_merges = max(num_merges)
    # round up to the nearest 50ls
    max_merges = max_merges + (bs - max_merges % bs)

    buckets = np.arange(min_merges, max_merges + bs, bs)
    # print (buckets)
    count = np.histogram(num_merges, bins=buckets)[0]
    num_merges = np.histogram(num_merges, bins=buckets)[1][:-1]

    num_merges = [str(x) for x in num_merges]
    # print (np.histogram(num_merges, bins=buckets))
    # print (num_merges, count)
    plt.xlabel('Number of merges')
    plt.ylabel('Normalized Frequency')

    plt.figure(figsize=(10, 6))

    plt.bar(num_merges, count/total_games, label='Number of Merges', color='skyblue')
 
    if title_suf == "":
        plt.title(f'Number of merges across {total_games} games')
    else :
        plt.title(f'Number of merges across {total_games} games - {title_suf}')
    plt.legend()
    plt.grid(axis='y')

    plt.savefig("./plots/num_merges/" + fname + ".png")
    plt.savefig("./plots/num_merges/" + fname + ".svg")
    plt.cla()

def hist_merge_scores(merge_scores, fname, title_suf = "", bs = 2000):
    print ("plotting merge_scores")
    # plots the merge scores per game -- merge_scores is an array scores
    total_games = len(merge_scores)
    # Create 'plots' directory if it doesn't exist
    if not os.path.isdir('plots/merge_scores'):
        os.mkdir('plots/merge_scores')

    merge_scores, count = np.unique(merge_scores, return_counts = True)

    min_scores = min(merge_scores)
    min_scores = min_scores - (min_scores % bs)
    max_scores = max(merge_scores)
    max_scores = max_scores + (bs - max_scores % bs)

    buckets = np.arange(min_scores, max_scores + bs, bs)
    count = np.histogram(merge_scores, bins=buckets)[0]
    merge_scores = np.histogram(merge_scores, bins=buckets)[1][:-1]

    merge_scores = [str(x) for x in merge_scores]
    plt.figure(figsize=(10, 6))
    plt.bar(merge_scores, count/total_games, label='Merge Scores', color='skyblue')
    plt.xlabel('Game')
    plt.ylabel('Merge Scores')
    if title_suf == "":
        plt.title(f'Merge Scores across {total_games} games')
    else: 
        plt.title(f'Merge Scores across {total_games} games - {title_suf}')
    plt.grid(axis='y')

    plt.legend()

    plt.savefig("./plots/merge_scores/" + fname + ".png")
    plt.savefig("./plots/merge_scores/" + fname + ".svg")
    plt.cla()

def hist_tiles(tiles, fname, title_suf = "", exponent = False):
    print ("plotting tiles")
    # plots the normalized dist. of tiles -- tiles is a list of lists, each list contains the tiles values for a game (2^x)
    if not os.path.isdir('plots/tiles_hits'):
        os.mkdir('plots/tiles_hits')
    
    total_games = len(tiles)
    flat_tiles = [item for sublist in tiles for item in sublist]
    
    if exponent:
        flat_tiles = [2**x for x in flat_tiles]
    
    tiles_values, count = np.unique(flat_tiles, return_counts = True)
    n_count = count/total_games

    tiles_values = [str(x) for x in tiles_values]
    
    plt.figure(figsize=(10, 6))
    plt.bar(tiles_values, n_count, color='skyblue')
    plt.xlabel('Tile Value')
    plt.ylabel('Normalized Frequency')
    if title_suf == "":
        plt.title(f'Normalized Distribution of Tile Values Over {total_games} games')
    else:
        plt.title(f'Normalized Distribution of Tile Values Over {total_games} games - {title_suf}')
    # plt.xticks(tiles_values)
    plt.grid(axis='y')

    plt.legend()

    plt.savefig("./plots/tiles_hits/" + fname + ".png")
    plt.savefig("./plots/tiles_hits/" + fname + ".svg")
    plt.cla()

def avg_branch(name, branches, scores):
    if not os.path.isdir('plots/avg_branch'):
        os.mkdir('plots/avg_branch')

    plt.plot(branches, scores)
    plt.title('Branch Factor vs Average Score')

    plt.savefig("./plots/avg_branch/" + name + ".png")
    plt.cla()