import copy
import random

import numpy as np
from numba import jit


@jit
def check_game(matrix):
    # If there is at least one empty square
    game_end = 0 not in matrix

    # If no empty square but you can still merge
    if game_end:
        for j in range(3):
            for k in range(3):
                if matrix[j][k] == matrix[j + 1][k] or matrix[j][k] == matrix[j][k + 1]:
                    game_end = False

        for j in range(3):
            if matrix[3][j] == matrix[3][j + 1]:
                game_end = False

        for j in range(3):
            if matrix[j][3] == matrix[j + 1][3]:
                game_end = False

    return game_end


class Game2048:
    def __init__(self):
        self.matrix = np.zeros((4, 4), dtype=np.int64)
        self.matrix[random.randint(0, 3)][random.randint(0, 3)] = random.choice([2, 4])
        self.game_end = False
        self.merge_score = 0

    def __str__(self):
        output = ""
        for row in self.matrix:
            output += str(row) + "\n"
        return output

    def game_reset(self):
        self.matrix = np.zeros((4, 4), dtype=np.int64)
        self.matrix[random.randint(0, 3)][random.randint(0, 3)] = random.choice([2, 4])
        self.game_end = False
        self.merge_score = 0
        
    def check_game(self):
        # If there is at least one empty square
        self.game_end = not (
            0 in self.matrix[0]
            or 0 in self.matrix[1]
            or 0 in self.matrix[2]
            or 0 in self.matrix[3]
        )

        # If no empty square but you can still merge
        if self.game_end:
            for j in range(3):
                for k in range(3):
                    if (
                        self.matrix[j][k] == self.matrix[j + 1][k]
                        or self.matrix[j][k] == self.matrix[j][k + 1]
                    ):
                        self.game_end = False

            for j in range(3):
                if self.matrix[3][j] == self.matrix[3][j + 1]:
                    self.game_end = False

            for j in range(3):
                if self.matrix[j][3] == self.matrix[j + 1][3]:
                    self.game_end = False

    def get_number(self):
        row, col = random.randint(0, 3), random.randint(0, 3)
        while self.matrix[row][col] != 0:
            row, col = random.randint(0, 3), random.randint(0, 3)

        self.matrix[row][col] = random.choice([2, 4])

    def rotate(self):
        self.matrix = np.rot90(self.matrix, k=3)

    def double_rotate(self):
        self.matrix = np.rot90(self.matrix, k=6)

    def merge(self):
        matrix_copy = copy.deepcopy(self.matrix)
        for col in range(len(self.matrix[0])):
            s = []
            for row in range(len(self.matrix)):
                if self.matrix[row][col] != 0:
                    s.append(self.matrix[row][col])
            i = 0
            while i < len(s) - 1:
                if s[i] == s[i + 1]:
                    s[i] *= 2
                    self.merge_score += s[i]
                    s.pop(i + 1)
                    i -= 1
                i += 1
            for row in range(len(self.matrix)):
                if len(s) > 0:
                    val = s.pop(0)
                    self.matrix[row][col] = val
                else:
                    self.matrix[row][col] = 0

        if (matrix_copy != self.matrix).any():
            self.get_number()
        self.check_game()

    def move_up(self):
        self.merge()

    def move_down(self):
        self.double_rotate()
        self.merge()
        self.double_rotate()

    def move_right(self):
        self.double_rotate()
        self.rotate()
        self.merge()
        self.rotate()

    def move_left(self):
        self.rotate()
        self.merge()
        self.double_rotate()
        self.rotate()

    def make_move(self, move):
        if move == 0:
            self.move_up()
        if move == 1:
            self.move_down()
        if move == 2:
            self.move_left()
        if move == 3:
            self.move_right()

    def get_sum(self):
        total_sum = 0
        for row in self.matrix:
            total_sum += sum(row)
        return total_sum

    def max_num(self):
        return max(map(max, self.matrix))

    def get_merge_score(self):
        return self.merge_score

    def get_corner_score(self):
        corner_score = 0
        max_tile = self.max_num()

        # # Define weights for each position
        # weights = [
        #     [10, 8, 7, 6],
        #     [8, 6, 5, 4],
        #     [7, 5, 3, 2],
        #     [6, 4, 2, 1]
        # ]

        # for i in range(4):
        #     for j in range(4):
        #         corner_score += self.matrix[i][j] * weights[i][j]

        # Give a bonus score if the max tile is in the corner
        max_tile_position = self.find_tile_position(max_tile)
        if (
            max_tile_position == (0, 0)
            or max_tile_position == (0, 3)
            or max_tile_position == (3, 0)
            or max_tile_position == (3, 3)
        ):
            corner_score += max_tile * 100

        return corner_score

    def find_tile_position(self, value):
        for i in range(4):
            for j in range(4):
                if self.matrix[i][j] == value:
                    return (i, j)
        return None
