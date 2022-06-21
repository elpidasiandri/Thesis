from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecMonitor
from envs import VecSudokuEnv

if __name__ == '__main__':
    env = VecMonitor(VecSudokuEnv())

    # policy_kwargs = {'net_arch': }

    model = A2C("CnnPolicy",
                env,
                tensorboard_log="Sudoku_A2C_CNN",
                verbose=1
                )

    i = 2
