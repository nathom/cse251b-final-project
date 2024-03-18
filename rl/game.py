import copy
import random
import numpy as np


# Game mechanics engine. Used by both the UI and the simulator.
class Game:
    def __init__(self, init_tile_matrix=None, init_score=0):
        #print("Making a new game")
        self.board_size = 4
        self.set_state(init_tile_matrix, init_score)
        self.num_merges = 0
        #print("Game made", self.tile_matrix)

    # set the game state using the given initialization state and total points
    def set_state(self, init_tile_matrix=None, init_score=0):
        self.undoMat = []
        self.score = init_score
        if init_tile_matrix is None:
            self.tile_matrix = self.new_tile_matrix()
            # print("new tile matrix", self.tile_matrix)
            self.place_random_tile()
            self.place_random_tile()
            # print("Added two tiles", self.tile_matrix)
            # exit()
        else:
            self.tile_matrix = copy.deepcopy(init_tile_matrix)
        self.board_size = len(self.tile_matrix)

        # print("set state", self.tile_matrix)

    def new_tile_matrix(self):
        return np.zeros((self.board_size, self.board_size), dtype = np.uint64).tolist()
        # return [[0] * self.board_size] * self.board_size

    # tuple representing the current game state
    def current_state(self):
        return (copy.deepcopy(self.tile_matrix), self.score)

    # performs a move in the specified direction and places a random tile
    def move_and_place(self, direction):
        # print(direction)
        if self.move(direction):
            self.place_random_tile()

    def rotate_matrix_clockwise(self): # anti-clockwise... 
        # print("rotating", self.tile_matrix)
        tm = self.tile_matrix
        for i in range(0, int(self.board_size / 2)):
            for k in range(i, self.board_size - i - 1):
                temp1 = tm[i][k] # top left
                temp2 = tm[self.board_size - 1 - k][i] # bottom left 
                temp3 = tm[self.board_size - 1 - i][self.board_size - 1 - k] # bottom right 
                temp4 = tm[k][self.board_size - 1 - i] # top right
                tm[self.board_size - 1 - k][i] = temp1 # new bottom left takes in top left -- it should take in bottom right 
                tm[self.board_size - 1 - i][self.board_size - 1 - k] = temp2 # new bottom right takes in bottom left -- it should take in top right
                tm[k][self.board_size - 1 - i] = temp3 # new top right takes in bottom right -- it should take in top left 
                tm[i][k] = temp4 # new top left takes in top right -- it should take in bottom left 
        # print("rotated", self.tile_matrix)

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
        tm = self.tile_matrix
        for i in range(0, self.board_size):
            for k in range(0, self.board_size - 1):
                if tm[i][k] == tm[i][k + 1] and tm[i][k] != 0:
                    tm[i][k] = tm[i][k] * 2
                    tm[i][k + 1] = 0
                    self.score += tm[i][k]
                    self.move_tiles()
                    self.num_merges +=1 

    def can_move(self):
        # print("in can move", self.tile_matrix)
        tm = self.tile_matrix
        for i in range(0, self.board_size):
            for j in range(1, self.board_size):
                # print("i", i, "j", j)
                if tm[i][j - 1] == 0 and tm[i][j] > 0:
                    return True
                elif (tm[i][j - 1] == tm[i][j]) and tm[i][j - 1] != 0:
                    return True
        # print("can't move??")
        return False

    def place_random_tile(self):
        while True:
            i = random.randint(0, self.board_size - 1)
            j = random.randint(0, self.board_size - 1)
            if self.tile_matrix[i][j] == 0:
                break   
        
        # print("bef", self.tile_matrix)
        # print("adding tile at", i, j)
        # print (self.tile_matrix[i], type(self.tile_matrix[i]))
        self.tile_matrix[i][j] = 2
        # print(self.tile_matrix)

    def undo(self):
        if len(self.undoMat) > 0:
            m = self.undoMat.pop()
            self.tile_matrix = m[0]
            self.score = m[1]

    def addToUndo(self):
        self.undoMat.append((copy.deepcopy(self.tile_matrix), self.score))

    def save_state(self, filename="savedata"):
        f = open(filename, "w")
        line = " ".join(
            [
                str(self.tile_matrix[int(x / self.board_size)][x % self.board_size])
                for x in range(0, self.board_size**2)
            ]
        )
        f.write(str(self.board_size) + " " + str(self.score) + " " + line)
        f.close()

    def load_state(self, filename="savedata"):
        f = open(filename, "r")
        self.load_state_line(f.readline())
        f.close()

    def load_state_line(self, line):
        split = line.split(" ")
        self.board_size = int(split[0])
        new_score = int(split[1])
        new_tm = self.new_tile_matrix()
        for i in range(0, self.board_size**2):
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
        # print("in game over", self.tile_matrix)
        found_dir = False
        for _ in range(0, 4):
            self.rotate_matrix_clockwise()
            if self.can_move():
                found_dir = True
        #print(found_dir)
        return not found_dir

    def max_num(self) -> int:
        return max([max(row) for row in self.tile_matrix])

    def get_sum(self) -> int:
        return sum([sum(row) for row in self.tile_matrix])

    def get_merge_score(self) -> int:
        return self.score
    
    def get_num_merges(self) -> int:
        return self.num_merges

    def get_tile_array(self):
        flattened = [item for sublist in self.tile_matrix for item in sublist]
        return flattened