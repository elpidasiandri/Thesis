from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor

from algorithms import DeviceAlternatingPPO
from envs import VecSudokuEnv
from policies.sudoku import SudokuACPolicy

model_name = "SudokuMixer"

if __name__ == '__main__':
    num_instances = 16

    env = VecMonitor(VecSudokuEnv(num_instances))

    model = DeviceAlternatingPPO(SudokuACPolicy,
                                 env,
                                 learning_rate=1e-4,
                                 n_steps=5120,
                                 batch_size=640 * num_instances,
                                 tensorboard_log="../bin",
                                 verbose=1
                                 )

    callbacks = [CheckpointCallback(20480, "../models_bin", model_name)]

    model.learn(total_timesteps=500_000_000,
                callback=callbacks,
                tb_log_name=model_name)
    model.save("models_bin/" + model_name + "_final")

    env.close()
