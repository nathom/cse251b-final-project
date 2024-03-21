import time
from datetime import timedelta

import click

import sys
sys.path.append('../')  # Add parent directory to the Python path

from .monte_carlo import monte_carlo_run
from .ai import expectimax_run
from data.metrics import *


@click.group()
def play():
    """Main command for playing various game strategies."""
    pass


@play.command()
@click.option(
    "--num-iters",
    type=int,
    default=100,
    help="Number of iterations for Monte Carlo simulation.",
)
@click.option(
    "--num-trials",
    type=int,
    default=100,
    help="Number of trials for Monte Carlo simulation.",
)
@click.option(
    "--eval-method",
    type=click.Choice(["sum", "largest", "merge"], case_sensitive=False),
    default="sum",
    help="Evaluation method for Monte Carlo simulation.",
)
def montecarlo(num_iters, num_trials, eval_method):
    """Command for playing with Monte Carlo strategy."""
    print("Playing with monte carlo strat")
    # Eval method 0: sum of all tiles
    # Eval method 1: largest tiles on board, then sum of all tiles if tied largest tile
    # Eval method 2: merge score
    max_val_results = [0] * num_trials
    total_sum_results = [0] * num_trials
    total_merge_score = [0] * num_trials
    method_map = {"sum": 0, "largest": 1, "merge": 2}
    method = method_map[eval_method]
    start_time = time.time()
    for i in range(num_trials):
        (
            max_val_results[i],
            total_sum_results[i],
            total_merge_score[i],
        ) = monte_carlo_run(num_iters, method)
    end_time = time.time()

    total_sum_avg = sum(total_sum_results) / num_trials
    max_val_avg = sum(max_val_results) / num_trials
    total_merge_avg = sum(total_merge_score) / num_trials

    print("total sum avg: " + str(total_sum_avg))
    print("max val avg: " + str(max_val_avg))
    print("merge score avg: " + str(total_merge_avg))
    print()
    print("time taken: ", str(timedelta(seconds=(end_time - start_time))))


@play.command()
def rl():
    """Command for playing with Reinforcement Learning strategy."""
    click.echo("Reinforcement Learning strategy selected.")


@play.command()
@click.option(
    "--num-trials",
    type=int,
    default=100,
    help="Number of trials for Expectimax.",
)
def expectimax(num_trials):
    """Command for playing with Expectimax strategy."""
    click.echo("Expectimax strategy selected.")
    max_val_results = [0] * num_trials
    total_sum_results = [0] * num_trials
    total_merge_score = [0] * num_trials
    num_merge = [0] * num_trials
    tile_array = [[0]] * num_trials
    start_time = time.time()
    print ("running expectimax")
    for i in range(num_trials):
        print("running trial: ", i+1, " of ", num_trials)
        (
            max_val_results[i], # largest tile
            total_sum_results[i], # sum of all tiles
            total_merge_score[i], # merge score
            num_merge[i], # number of merges
            tile_array[i] # losing configuration
        ) = expectimax_run()
    end_time = time.time()
    print("done running expectimax")
    total_sum_avg = sum(total_sum_results) / num_trials
    max_val_avg = sum(max_val_results) / num_trials
    total_merge_avg = sum(total_merge_score) / num_trials

    print("total sum avg: " + str(total_sum_avg))
    print("max val avg: " + str(max_val_avg))
    print("merge score avg: " + str(total_merge_avg))
    print()
    print("time taken: ", str(timedelta(seconds=(end_time - start_time))))
    
    fname =  "expectimax_" + str(num_trials) + "_trials"
    title = "expectimax"
    # dump all the results to a file 
    with open(fname + ".csv", "w") as f:
    # write the header:  "Game Number,Number of Moves,Score,Largest Tile,Sum of Tiles, Number of Merges, Losing Configuration,seconds\n"
        f.write("Game Number, Merge Score, Largest Tile,Sum of Tiles, Number of Merges, Losing Configuration\n")
        for i in range(num_trials):
            f.write(str(i) + "," + str(total_merge_score[i]) + "," + str(max_val_results[i]) + "," + str(total_sum_results[i]) + "," + str(num_merge[i]) + "," + str(tile_array[i]) + "\n") 
    
    hist_max_val(max_val_results,fname, title_suf = title)
    hist_num_merges(num_merge,fname,title_suf = title, bs = 100)
    hist_merge_scores(total_merge_score, fname, title_suf = title, bs = 2000)
    hist_tiles(tile_array, fname, title_suf = title)


def main():
    play()
