from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random

class ShowSudokuEnv(Env):
    def __init__(self):
        # Actions we can take, down, stay, up
        self.action_space = Discrete(9)
        # Sudoku array
        # TODO: Notice the changes I made. Otherwise you did good.
        #  9x9 table, 9 squares in total. Lows and highs are good.
        self.observation_space = Box(low=np.array([1, 1]), high=np.array([9, 9]))
        # Set start sudoku
        # FIXME: state represents the current sudoku table, it is not a number
        self.state = np.empty((9, 9)) #kenos pinakas 9x9
        self.state[:] = np.NaN #ola ta stoixeia ginontai NaN

    def step(self, action):
        # Apply action
        # 0 -1 = -1 temperature
        # 1 -1 = 0
        # 2 -1 = 1 temperature
        # FIXME: the statements below return a boolean and have no effect. Check your PyCharm warnings.
        #  What you need is an assignment on the Sudoku table, which is the state.
        np.array([self.state], _) != action
        np.array(_, [self.state]) != action
        # Reduce shower length by 1 second
        self.shower_length -= 1

        # Calculate reward
        # FIXME: there is no stateline declared anywhere
        if np.array([self.stateline], [self.state]).size == 1:
            reward = 1
        else:
            reward = -1

            # Check if shower is done
        if self.shower_length <= 0:
            done = True
        else:
            done = False

        # Set placeholder for info
        info = {}

        # Return step information
        return self.state, reward, done, info

    def render(self, **kwargs):
        # Implement viz
        pass

    def reset(self):
        # Reset shower temperature
        self.state = 38 + random.randint(-3, 3)
        # Reset shower time
        self.shower_length = 60
        return self.state

