import numpy as np


def transfer(chessboard):
    onehotboard = np.zeros((16,) + chessboard.shape)
    onehotboard[0, chessboard == 0] = 1
    for i in range(1, 16):
        onehotboard[i, chessboard == 2 ** i] = 1

    return onehotboard
