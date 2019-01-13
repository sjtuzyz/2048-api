from game2048.game import Game
from game2048.agents import TrainAgent
from game2048.displays import Display
import torch

GAME_SIZE = 4
SCORE_TO_WIN = 2048
Limit=2000
PATH = 'model2048.pkl'
SAVE_PATH = 'model2048.pkl'

game = Game(GAME_SIZE, SCORE_TO_WIN, random=False)
agent = TrainAgent(game,True,display=Display(), train=True,path=PATH)
for i in range(Limit):
    agent.play(verbose=False)
    print("Game: {} Score: {}".format(i, agent.game.score))
    agent.new_game(game=Game(GAME_SIZE, SCORE_TO_WIN, random=False))
    if (i+1)%100 == 0:
        torch.save(agent.net.state_dict(),SAVE_PATH)
