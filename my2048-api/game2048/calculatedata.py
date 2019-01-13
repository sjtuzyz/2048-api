import numpy as np
from .onehot_board import transfer


def caltraindata(board, direction):
    train_data = np.zeros((6, 16) + board.shape)
    expdirection = np.zeros([6, 1])

    for i in range(0, 4):
        train_data[i, :, :, :] = transfer(np.rot90(board, i))
        expdirection[i] = (direction + i) % 4

    row_map = {0: 2, 1: 1, 2: 0, 3: 3}
    col_map = {0: 0, 1: 3, 2: 2, 3: 1}

    train_data[4, :, :, :] = transfer(board[:, -1::-1])
    expdirection[4] = row_map[direction]

    train_data[5, :, :, :] = transfer(board[-1::-1, :])
    expdirection[5] = col_map[direction]

    return train_data, expdirection
