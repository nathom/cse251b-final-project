from __future__ import absolute_import, division, print_function

import math

from .game import Game

MOVES = {0: "up", 1: "left", 2: "down", 3: "right"}
MAX_PLAYER, CHANCE_PLAYER = 0, 1


# Tree node. To be used to construct a game tree.
class Node:
    # Recommended: do not modify this __init__ function
    def __init__(self, state, player_type):
        self.state = (state[0], state[1])

        # to store a list of (direction, node) tuples
        self.children = []

        self.player_type = player_type

    # returns whether this is a terminal state (i.e., no children)
    def is_terminal(self):
        isEmpty = len(self.children) == 0
        return isEmpty


# AI agent. Determine the next move.
class AI:
    def __init__(self, root_state, search_depth=3):
        self.root = Node(root_state, MAX_PLAYER)
        self.search_depth = search_depth
        self.simulator = Game(*root_state)

    def count_open_tiles(self, matrix):
        count = 0
        for row in matrix:
            count += row.count(0)
        return count

    def count_mergeable(self, matrix):
        n = len(matrix)
        merge_count = 0
        for i in range(n):
            for j in range(n):
                if j < n - 1 and matrix[i][j] == matrix[i][j + 1]:
                    # Cells are adjacent horizontally
                    merge_count += 1
                elif i < n - 1 and matrix[i][j] == matrix[i + 1][j]:
                    # Cells are adjacent vertically
                    merge_count += 1
                elif (
                    j < n - 2
                    and matrix[i][j] == matrix[i][j + 2]
                    and all(elem == 0 for elem in matrix[i][j + 1 : j + 2])
                ):
                    # Cells are in the same row with a zero cell in between
                    merge_count += 1
                elif (
                    i < n - 2
                    and matrix[i][j] == matrix[i + 2][j]
                    and all(elem == 0 for elem in [matrix[i + 1][j], matrix[i + 2][j]])
                ):
                    # Cells are in the same column with a zero cell in between
                    merge_count += 1
        return merge_count

    def large_on_edges(self, matrix):
        n = len(matrix)
        count = 0
        # First and last rows
        for i in range(n):
            if matrix[0][i] >= 16:
                count += 1
            if matrix[0][i] >= 32:
                count += 1
            if matrix[0][i] >= 64:
                count += 1
            if matrix[0][i] >= 128:
                count += 1
            if matrix[n - 1][i] >= 16:
                count += 1
            if matrix[n - 1][i] >= 32:
                count += 1
            if matrix[n - 1][i] >= 64:
                count += 1
            if matrix[n - 1][i] >= 128:
                count += 1
        # First and last columns (not double counting corners)
        for i in range(1, n - 1):
            if matrix[i][0] >= 16:
                count += 1
            if matrix[i][0] >= 32:
                count += 1
            if matrix[i][0] >= 64:
                count += 1
            if matrix[i][0] >= 128:
                count += 1
            if matrix[i][n - 1] >= 16:
                count += 1
            if matrix[i][n - 1] >= 32:
                count += 1
            if matrix[i][n - 1] >= 64:
                count += 1
            if matrix[i][n - 1] >= 128:
                count += 1
        return count

    # Potential heuristic functions
    # Bonus for large tiles on the edges
    # Bonus for more open tiles
    def heuristic_build_tree(self, node=None, depth=0):
        if depth == 0:
            return
        else:
            # If it is the player's turn
            if node.player_type == MAX_PLAYER:
                # Save the current matrix and score
                curMatrix, curScore = self.simulator.current_state()
                # For every possible move
                for i in MOVES:
                    # Set the simulator to the new state
                    self.simulator.set_state(curMatrix, curScore)

                    # Move the simulator appropriately
                    self.simulator.move(i)

                    # Save matrix and score of particular move
                    moveMatrix, moveScore = self.simulator.current_state()

                    open_bonus = self.count_open_tiles(moveMatrix)

                    edge_bonus = self.large_on_edges(moveMatrix)

                    mergeable_bonus = self.count_mergeable(moveMatrix)

                    val = moveScore + 5 * edge_bonus + 15 * mergeable_bonus

                    # Create a new node for the possible move
                    curNode = Node((moveMatrix, val), CHANCE_PLAYER)

                    if moveMatrix != curMatrix:
                        # Add the node the parent's list of children
                        node.children.append((i, curNode))
                        # Build the tree of each node
                        self.heuristic_build_tree(curNode, depth - 1)

                    # Reset simulator to original state
                    self.simulator.set_state(curMatrix, curScore)

            # If it is the RNG turn
            elif node.player_type == CHANCE_PLAYER:
                # Save the current matrix and score
                curMatrix, curScore = self.simulator.current_state()
                # Find every possible location matrix
                locations = []
                length = len(curMatrix)
                for i in range(length):
                    for j in range(length):
                        if curMatrix[i][j] == 0:
                            newMatrix = [row[:] for row in curMatrix]
                            newMatrix[i][j] = 2
                            locations.append(newMatrix)

                # For every possible location
                for i in locations:
                    # Set the simulator to the new state
                    # self.simulator.set_state(i, curScore + 2)
                    self.simulator.set_state(i, curScore)
                    # Create a new node for each possible location
                    curNode = Node(self.simulator.current_state(), MAX_PLAYER)
                    # Add the node the parent's list of children
                    node.children.append((None, curNode))
                    # Build the tree of each node
                    self.heuristic_build_tree(curNode, depth - 1)
                    # Reset simulator to original state
                    self.simulator.set_state(curMatrix, curScore)

    def build_tree(self, node: Node, depth=0):
        if depth == 0:
            return
        # If it is the player's turn
        if node.player_type == MAX_PLAYER:
            # Save the current matrix and score
            curMatrix, curScore = self.simulator.current_state()
            # For every possible move
            for i in MOVES:
                # Set the simulator to the new state
                self.simulator.set_state(curMatrix, curScore)

                # Move the simulator appropriately
                self.simulator.move(i)

                # Save matrix and score of particular move
                moveMatrix, _ = self.simulator.current_state()

                # Create a new node for the possible move
                curNode = Node(self.simulator.current_state(), CHANCE_PLAYER)

                # If matrix did not change is not a valid move
                if moveMatrix != curMatrix:
                    # Add the node the parent's list of children
                    node.children.append((i, curNode))
                    # Build the tree of each node
                    self.build_tree(curNode, depth - 1)

                # Reset simulator to original state
                self.simulator.set_state(curMatrix, curScore)

        # If it is the RNG turn
        elif node.player_type == CHANCE_PLAYER:
            # Save the current matrix and score
            curMatrix, curScore = self.simulator.current_state()
            # Find every possible location matrix
            locations = []
            length = len(curMatrix)
            for i in range(length):
                for j in range(length):
                    if curMatrix[i][j] == 0:
                        newMatrix = [row[:] for row in curMatrix]
                        newMatrix[i][j] = 2
                        locations.append(newMatrix)

            # For every possible location
            for i in locations:
                # Set the simulator to the new state
                # self.simulator.set_state(i, curScore + 2)
                self.simulator.set_state(i, curScore)
                # Create a new node for each possible location
                curNode = Node(self.simulator.current_state(), MAX_PLAYER)
                # Add the node the parent's list of children
                node.children.append((None, curNode))
                # Build the tree of each node
                self.build_tree(curNode, depth - 1)
                # Reset simulator to original state
                self.simulator.set_state(curMatrix, curScore)

    def chance(self, node):
        matrix = node.state[0]
        length = len(matrix)
        emptyCells = 0
        for i in range(length):
            for j in range(length):
                if matrix[i][j] == 0:
                    emptyCells += 1
        chance = 1 / (emptyCells + 1)
        return chance

    # TODO: expectimax calculation.
    # Return a (best direction, expectimax value) tuple if node is a MAX_PLAYER
    # Return a (None, expectimax value) tuple if node is a CHANCE_PLAYER
    def expectimax(self, node=None):
        # print("here", node)
        if node.is_terminal():
            return None, node.state[1]
        # If player
        elif node.player_type == MAX_PLAYER:
            # print("here ?")
            bestValue = -math.inf
            bestMove = None
            for n in node.children:
                curMove, node = n
                _, curValue = self.expectimax(node)
                if curValue > bestValue:
                    bestValue = curValue
                    bestMove = curMove
            # print("best move", bestMove)
            return bestMove, bestValue
        # If RNG move
        elif node.player_type == CHANCE_PLAYER:
            value = 0
            for n in node.children:
                _, node = n
                curMove, curValue = self.expectimax(node)
                value = value + self.chance(node) * curValue
            return None, value

    # Return decision at the root
    def compute_decision(self):
        # print ("computing decision")
        self.build_tree(self.root, self.search_depth)
        # print("built tree")
        direction, _ = self.expectimax(self.root)
        # print("direction", direction)
        return direction

    # TODO (optional): implement method for extra credits
    def compute_decision_ec(self):
        self.heuristic_build_tree(self.root, self.search_depth)
        direction, _ = self.expectimax(self.root)
        return direction
    
def expectimax_run():
    game = Game()
    i = 0 
    # print("starting game")
    while not game.game_over():
        # print("Iteration: ", i, "max num", game.max_num())
        ai = AI(game.current_state())
        direction = ai.compute_decision()
        # MOVES = {0: "up", 1: "left", 2: "down", 3: "right"}
        if direction != None:
            game.move_and_place(direction)
        i += 1
    # print("Game over")
    return game.max_num(), game.get_sum(), game.get_merge_score(), game.get_num_merges(), game.get_tile_array()
