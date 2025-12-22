"""
implementing the training / iterative improvement algorithm
"""
import math
import torch
from model import BasicNet
from state import Game
from collections import defaultdict

MCTS = 800 # same as paper
C_ULT = 0.3 # tradeoff between eploitation, exploration

class AlphaZero:

    def __init__(self):

        # we keep the following information for each edge (s -> a), representing transition from state s via action a
        self.prior: defaultdict[int, defaultdict[int, float]] = defaultdict(defaultdict) # probability of move a being good choice from state s
        self.visits: defaultdict[int, defaultdict[int, int]] = defaultdict(defaultdict) # times we have chosen action a from state s
        self.q: defaultdict[int, defaultdict[int, float]] = defaultdict(defaultdict) # value of state created from s to a

        self.total: int = 0     # number of total visits 


    def uct(self, s, a):
        return self.q[s][a] + C_ULT * self.prior[s][a] * (math.sqrt(self.total) / (1 + self.visits[s][a]))


    def value(self, game: Game):
        raise NotImplementedError


    def get_best_move(self, game: Game):
        move_values = {}
        for move in game.get_valid_moves():
            game.make_move(move)
            move_values[move] = self.value(game)
            game.undo_move()
        
        return max(move_values, key=move_values.get)
