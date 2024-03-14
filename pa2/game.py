# NOTE: Do not modify.
import copy, random


# Game mechanics engine. Used by both the UI and the simulator.
class Game2048():
    def __init__(self, init_tile_matrix = None, init_score = 0):
        self.board_size = 4
        self.game_end = False
        self.set_state(init_tile_matrix, init_score)

    # set the game state using the given initialization state and total points
    def set_state(self, init_tile_matrix = None, init_score = 0):
        self.undoMat = []
        self.score = init_score
        self.num_merges = 0
        if init_tile_matrix == None:
            self.tile_matrix = self.new_tile_matrix()
            self.place_random_tile()
            self.place_random_tile()
        else:
            self.tile_matrix = copy.deepcopy(init_tile_matrix)
        self.board_size = len(self.tile_matrix)

    def new_tile_matrix(self):
        return [[0 for i in range(self.board_size)] for j in range(self.board_size)]

    # tuple representing the current game state
    def current_state(self):
        return (copy.deepcopy(self.tile_matrix), self.score)

    # performs a move in the specified direction and places a random tile
    def move_and_place(self, direction):
        if self.move(direction):
            self.place_random_tile()

    def rotate_matrix_clockwise(self):
        tm = self.tile_matrix
        for i in range(0, int(self.board_size/2)):
            for k in range(i, self.board_size- i - 1):
                temp1 = tm[i][k]
                temp2 = tm[self.board_size - 1 - k][i]
                temp3 = tm[self.board_size - 1 - i][self.board_size - 1 - k]
                temp4 = tm[k][self.board_size - 1 - i]
                tm[self.board_size - 1 - k][i] = temp1
                tm[self.board_size - 1 - i][self.board_size - 1 - k] = temp2
                tm[k][self.board_size - 1 - i] = temp3
                tm[i][k] = temp4

    # moves in the specified direction
    def move(self, direction):
        moved = False
        self.addToUndo()
        for i in range(0, direction):
            self.rotate_matrix_clockwise()
        if self.can_move():
            self.move_tiles()
            self.merge_tiles()
            moved = True
        for j in range(0, (4 - direction) % 4):
            self.rotate_matrix_clockwise()
        return moved

    def move_tiles(self):
        tm = self.tile_matrix
        for i in range(0, self.board_size):
            for j in range(0, self.board_size - 1):
                while tm[i][j] == 0 and sum(tm[i][j:]) > 0:
                    for k in range(j, self.board_size - 1):
                        tm[i][k] = tm[i][k + 1]
                    tm[i][self.board_size - 1] = 0

    def merge_tiles(self):
        # create a variable that increases everytime merge tiles is called
        tm = self.tile_matrix
        for i in range(0, self.board_size):
            for k in range(0, self.board_size - 1):
                if tm[i][k] == tm[i][k + 1] and tm[i][k] != 0:
                    tm[i][k] = tm[i][k] * 2
                    tm[i][k + 1] = 0
                    self.score += tm[i][k]
                    self.move_tiles()
                    self.num_merges+=1

    def can_move(self):
        tm = self.tile_matrix
        for i in range(0, self.board_size):
            for j in range(1, self.board_size):
                if tm[i][j-1] == 0 and tm[i][j] > 0:
                    return True
                elif (tm[i][j-1] == tm[i][j]) and tm[i][j-1] != 0:
                    return True
        return False

    def place_random_tile(self):
        while True:
            i = random.randint(0,self.board_size-1)
            j = random.randint(0,self.board_size-1)
            if self.tile_matrix[i][j] == 0:
                break
        self.tile_matrix[i][j] = 2

    def undo(self):
        if len(self.undoMat) > 0:
            m = self.undoMat.pop()
            self.tile_matrix = m[0]
            self.score = m[1]

    def addToUndo(self):
        self.undoMat.append((copy.deepcopy(self.tile_matrix),self.score))

    def save_state(self, filename="savedata"):
        f = open(filename, "w")
        line = " ".join([str(self.tile_matrix[int(x / self.board_size)][x % self.board_size])
                        for x in range(0, self.board_size**2)])
        f.write(str(self.board_size) + " " + str(self.score) + " " + line)
        f.close()

    def load_state(self, filename="savedata"):
        f = open(filename, "r")
        self.load_state_line(f.readline())
        f.close()

    def load_state_line(self,line):
        split = line.split(' ')
        self.board_size = int(split[0])
        new_score = int(split[1])
        new_tm = self.new_tile_matrix()
        for i in range(0, self.board_size ** 2):
            new_tm[int(i / self.board_size)][i % self.board_size] = int(split[2 + i])
        self.set_state(new_tm, new_score)

    # returns a list of all open (value 0) tiles
    def get_open_tiles(self):
        tiles = []
        for i in range(0, self.board_size):
            for j in range(0, self.board_size):
                if self.tile_matrix[i][j] == 0:
                    tiles.append((i, j))
        return tiles

    def game_over(self):
        found_dir = False
        for i in range(0, 4):
            self.rotate_matrix_clockwise()
            if self.can_move():
                found_dir = True

        self.game_end = not found_dir
        return not found_dir

    # WARNING: Deprecated: do not call this function in ai.py
    def get_state(self):
        return (self.tile_matrix, self.score)

    # WARNING: Deprecated: do not call this function in ai.py
    def reset(self, init_tile_matrix = None, init_score = 0):
        self.undoMat = []
        self.score = init_score
        if init_tile_matrix == None:
            self.tile_matrix = self.new_tile_matrix()
            self.place_random_tile()
            self.place_random_tile()
        else:
            self.tile_matrix = copy.deepcopy(init_tile_matrix)
        self.board_size = len(self.tile_matrix)

    def get_sum(self):
        total_sum = 0
        # print(self.tile_matrix)
        for row in self.tile_matrix:
            total_sum += sum(row)
        return total_sum

    def max_num(self):
        return max(map(max, self.tile_matrix))

    def get_merge_score(self):
        return self.score
    
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
        if max_tile_position == (0, 0) or max_tile_position == (0, 3) or max_tile_position == (3, 0) or max_tile_position == (3, 3):
            corner_score += max_tile * 100

        return corner_score

    def get_num_merges(self):
        # return the number of merges done during the game
        return self.num_merges
    
    def find_tile_position(self, value):
        for i in range(4):
            for j in range(4):
                if self.tile_matrix[i][j] == value:
                    return (i, j)
        return None
    
    def get_tiles_values(self):
        # return a list of all the values in the matrix
        v = [value for row in self.tile_matrix for value in row]
        # print (v)
        return v