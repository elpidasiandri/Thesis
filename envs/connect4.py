import numpy as np
from gym import Env
from gym.spaces import Box, Discrete
from scipy.signal import convolve2d


# TODO: Complete this

class Connect4Env(Env):

    def __init__(self,
                 width: int = 7,
                 height: int = 6,
                 connect: int = 4,
                 num_players: int = 2):
        self.width = width
        self.height = height
        self.connect = connect
        self.num_players = num_players

        self.action_space = Discrete(width)
        self.observation_space = Box(low=0, high=2, shape=[1, height, width])

        self.initial_state = np.zeros(self.observation_space.shape)

        self._build_winning_convolutions()

        self.state = self.initial_state.copy()
        self.switch = False
        self.winner = None

    def _build_winning_convolutions(self):
        self.convolutions = []
        self.convolutions.append(np.array([[1] * self.connect]))
        self.convolutions.append(np.array([[1]] * self.connect))
        self.convolutions.append(np.identity(self.connect))
        self.convolutions.append(np.diag(np.ones(self.connect)))

    def _check_win(self):
        for conv in self.convolutions:
            if np.any(convolve2d(self.state, conv, mode="valid") /
                      (self.connect * (1 + self.switch)) == 1):
                self.winner = 1 + self.switch
                return True
        return False

    def step(self, action):
        if self.switch:
            player = 1
        else:
            player = 0

        self.state[0, action, action] = player
        self.switch = not self.switch

        done = self._check_win()

        return self.state, 0, done, {}

    def reset(self):
        self.state = self.initial_state.copy()
        self.switch = False
        self.winner = None

    def render(self, mode="human"):
        pass
