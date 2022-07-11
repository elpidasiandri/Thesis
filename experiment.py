from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor

from algorithms import DeviceAlternatingPPO
from envs import VecSudokuEnv
from policies import SudokuACPolicy, ActorCriticCNNPolicy

if __name__ == '__main__':
    num_instances = 16

    env = VecMonitor(VecSudokuEnv(num_instances))

    model = DeviceAlternatingPPO(ActorCriticCNNPolicy,
                                 env,
                                 n_steps=5120,
                                 batch_size=640 * num_instances,
                                 tensorboard_log="bin",
                                 verbose=1
                                 )

    callbacks = [CheckpointCallback(20480, "models", "CNN_3layer")]

    model.learn(total_timesteps=500_000_000,
                callback=callbacks,
                tb_log_name="CNN_3layer")
    model.save("models/CNN_3layer_final")

    env.close()
