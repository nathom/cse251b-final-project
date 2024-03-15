import copy
import random

from .game_2048 import Game2048

NUM_ITERS = 100
NUM_TRIALS = 100
EVAL_METHOD = 0


def random_run(game: Game2048):
    game_copy = copy.deepcopy(game)
    while not game_copy.game_end:
        move = random.randint(0, 3)
        game_copy.make_move(move)

    return game_copy.get_sum(), game_copy.max_num(), game_copy.get_merge_score()


def monte_carlo_iter(game: Game2048, num_iters, eval_method):
    total_score = [0, 0, 0, 0]

    # For each move (0 - 3)
    for move in range(0, 4):
        game_copy = copy.deepcopy(game)
        game_copy.make_move(move)
        if str(game_copy) == str(game):
            continue

        # Try lots of paths with that move using random rollout policy
        for _ in range(num_iters):
            output = random_run(game_copy)  # 0 for largest tile, 1 for sum

            # Eval Method 0: Best total sum
            if eval_method == 0:
                total_score[move] += output[0]

            # Eval Method 1: Largest tile number
            elif eval_method == 1:
                total_score[move] += output[1]

            # Eval Method 2: Largest total merge score
            elif eval_method == 2:
                total_score[move] += output[2]

    best_move = total_score.index(max(total_score))
    game.make_move(best_move)

    # print(game)
    # print(game.max_num())
    # print(game.get_sum())
    # print(game.get_merge_score())
    # print("--------------------")


def monte_carlo_run(num_iters, eval_method):
    game = Game2048()
    # print("post construction", game.tile_matrix, "end")
    # print("game over", game.game_over())
    i = 0
    while not game.game_end:
        print("Iteration: ", i, "max num", game.max_num())
        monte_carlo_iter(game, num_iters, eval_method)

        i += 1

    # print("Max Square Value: {}".format(game.max_num()))
    # print("Total Square Sum: {}".format(game.get_sum()))
    # print("Total Merge Score: {}".format(game.get_merge_score()))
    return game.max_num(), game.get_sum(), game.get_merge_score()
