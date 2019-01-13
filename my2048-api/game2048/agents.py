import numpy as np
import torch
import torch.nn
from .expectimax import board_to_move
from .onehot_board import transfer
from .CNN import nn2048
from .calculatedata import caltraindata
import time


class Agent:
    '''Agent Base.'''

    def __init__(self, game, display=None):
        self.game = game
        self.display = display

    def play(self, max_iter=np.inf, verbose=False):
        n_iter = 0
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            self.game.move(direction)
            n_iter += 1
            if verbose:
                print("Iter: {}".format(n_iter))
                print("======Direction: {}======".format(
                    ["left", "down", "right", "up"][direction]))
                if self.display is not None:
                    self.display.display(self.game)

    def step(self):
        direction = int(input("0: left, 1: down, 2: right, 3: up = ")) % 4
        return direction


class RandomAgent(Agent):

    def step(self):
        direction = np.random.randint(0, 4)
        return direction


class ExpectiMaxAgent(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        from .expectimax import board_to_move
        self.search_func = board_to_move

    def step(self):
        direction = self.search_func(self.game.board)
        return direction


# self.state=torch.save(self.net.state_dict(),TrainPath)
TrainPath = 'model2048.pkl'
TestPath = 'model2048.pkl'


class TrainAgent(Agent):

    def __init__(self, game, load, display=None, train=True, path=None):
        super().__init__(game, display)
        self.train = train
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if path == None:
            path = TrainPath
        else:
            pass
        if train:
            if load == True:
                self.net = nn2048().to(self.device)
                self.net.load_state_dict(torch.load(path, map_location=self.device))
                self.lossfunction = torch.nn.CrossEntropyLoss()
                self.optimizer = torch.optim.SGD(self.net.parameters(), lr=1e-3, momentum=0.9)
            else:
                self.net = nn2048().to(self.device)


        else:
            if load == True:
                self.net = nn2048().to(self.device)
                self.net.load_state_dict(torch.load(path, map_location=self.device))
                self.net.eval()
            else:
                self.net = nn2048().to(self.device)

    def trainning(self, board, expdirection):
        train_data, train_targets = caltraindata(board, expdirection)
        train_data = torch.Tensor(train_data).to(self.device).float()
        train_targets = torch.Tensor(train_targets).to(self.device).long().squeeze(1)

        y = self.net.forward(train_data)
        loss = self.lossfunction(y, train_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def step(self):
        onehotboard = transfer(self.game.board)
        if self.train:
            direction = self.net.next(torch.Tensor(onehotboard.reshape(1, *onehotboard.shape)).to(self.device).float())
            expdirection = board_to_move(self.game.board)
            self.trainning(self.game.board, expdirection)

        else:
            direction = self.net.next(torch.Tensor(onehotboard.reshape(1, *onehotboard.shape)).to(self.device).float())
        return direction

    def new_game(self, game):
        self.game = game


class TestAgent(Agent):
    def __init__(self, game,display=None):
        super().__init__(game, display)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.TestPath = 'model2048.pkl'
        self.net = nn2048().to(self.device)
        self.net.load_state_dict(torch.load(self.TestPath, map_location=self.device))
        self.net.eval()

    def step(self):
        board = self.game.board
        onehotboard = transfer(board)
        time_start = time.time()
        direction = self.net.next(torch.Tensor(onehotboard.reshape(1, *onehotboard.shape)).to(self.device).float())
        direction = direction.item()
        time_end = time.time()
        time_cost = time_end - time_start
        print('time cost is', time_cost, 's')
        return direction

    def new_game(self, game):
        self.game = game
