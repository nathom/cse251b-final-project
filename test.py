import numpy as np


def rotate(matrix):
    out = []
    for col in range(len(matrix[0])):
        temp = []
        for row in reversed(range(len(matrix[0]))):
            temp.append(matrix[row][col])
        out.append(temp)
    return out


a = [list(range(4 * i, 4 * (i + 1))) for i in range(4)]
print(a)
print(rotate(a))
a = np.array(a)
print(a)
print(a.T)
print(np.rot90(a, k=3))
print(np.rot90(np.rot90(np.rot90(a))))
print(np.flip(a))
