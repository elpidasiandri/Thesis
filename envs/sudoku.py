import sys
from functools import partial
from itertools import product

import numpy as np
from gym import Env
from gym.spaces import MultiDiscrete, Box
from stable_baselines3.common.vec_env import DummyVecEnv


class SudokuEnv(Env):
    _win = 0
    _unfinished = 1
    _loss = 2

    _initial_state = np.empty((9, 9))
    _initial_state[:] = np.NaN

    def __init__(self):
        self.action_space = MultiDiscrete([9, 9, 9])  # row, column, number
        self.observation_space = Box(low=np.array([1, 1]), high=np.array([9, 9]))
        self.state = self._initial_state.copy()

    def _check_status(self) -> int:
        nan_mask = np.isnan(self.state)

        # validity check

        # non-validity = loss
        # row
        for i in range(self.state.shape[0]):
            r = self.state[i][~nan_mask[i]]
            if np.unique(r).size < r.size:
                return self._loss
        # column
        for i in range(self.state.shape[1]):
            c = self.state[:, i][~nan_mask[:, i]]
            if np.unique(c).size < c.size:
                return self._loss
        # sub-grids
        for i, j in product(*[[0, 3, 6]] * 2):
            subgrid = self.state[i:i + 3, j:j + 3][~nan_mask[i:i + 3, j:j + 3]]
            if np.unique(subgrid).size < subgrid.size:
                return self._loss

        # validity + all-filled = win
        if not nan_mask.sum():
            return self._win

        # validity = unfinished
        return self._unfinished

    def step(self, action: np.ndarray):
        """
        :param action: Multi-discrete action within ranges [0, 0, 0] and [8, 8, 8]
        :return: observation numpy array, reward scalar, episode done boolean, info dict
        """
        old_state = self.state.copy()

        # case: position already filled
        if not np.isnan(old_state[action[0], action[1]]):
            return old_state, -1, False, None

        # case: valid action
        self.state[action[0], action[1]] = action[2] + 1  # plus one for range 1-9

        status = self._check_status()

        # valid action sub-cases: win, unfinished, loss
        if status == self._win:
            return self.state.copy(), 10, True, None
        elif status == self._unfinished:
            return self.state.copy(), 1, False, None
        elif status == self._loss:
            return self.state.copy(), -5, True, None

    def render(self, **kwargs):
        state_list = self.state.tolist()

        sys.stdout.write('\n---------------\n')
        for i in range(len(state_list)):
            sys.stdout.write('|')
            r = state_list[i]
            for j in range(len(r)):
                sys.stdout.write(r[j])
                if (j + 1) % 3 == 0:
                    sys.stdout.write('|')
            if (i + 1) % 3 == 0:
                sys.stdout.write('\n---------------\n')

    def reset(self):
        self.state = self._initial_state.copy()
        return self.state.copy()


class VecSudokuEnv(DummyVecEnv):
    def __init__(self, n_envs=8):
        super(VecSudokuEnv, self).__init__([partial(SudokuEnv) for _ in range(n_envs)])
