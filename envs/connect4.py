import numpy as np
from gym import Env
from gym.spaces import Box, Discrete
from scipy.signal import convolve2d


# TODO: Complete this

class Connect4Env(Env):

    _win = 0
    _loss = 1
    _invalid = 2

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
        self.done = False

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
                return True
        return False

    def step(self, column):
        # TODO: this is really really wrong. There are two different rollouts.
        #  I need to use PettingZoo for a multi-agent environment:
        #  https://www.pettingzoo.ml/environment_creation#example-custom-environment
        #  https://www.pettingzoo.ml/wrappers#included-multi-agent-only-functions
        if self.done:
            return self.state, -20, True, {}

        player = self.switch + 1
        row = np.argmax(self.state[0, :, column] == 0)

        # case: column is full
        if self.state[0, row, column] != 0:
            # TODO: fix state depending on player
            return self.state, -1, False, {}

        self.state[0, row, column] = player

        # case: column is valid
        self.done = self._check_win()
        rew = 20 if self.done else 1

        self.switch = not self.switch
        # TODO: fix state depending on player
        return self.state, rew, False, {}

    def reset(self):
        self.state = self.initial_state.copy()
        self.switch = False
        self.done = False

    def render(self, mode="human"):
        pass
