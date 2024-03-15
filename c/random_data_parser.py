##
## PURPOSE: FOR RANDOM RUN OF GAME
##
import sys

scores = []
max_tiles = []
count_2048 = 0

with open(sys.argv[1], 'r') as file:
    for line in file:
        score, max_tile = line.strip().split(',')
        scores.append(int(score))
        max_tiles.append(int(max_tile))

        if int(max_tile) == 2048:
            count_2048 += 1

print("Avg Scores:", sum(scores)/len(scores))
print("Avg Max Tiles:", sum(max_tiles)/len(max_tiles))
print("2048 Tiles:", count_2048/len(max_tiles))