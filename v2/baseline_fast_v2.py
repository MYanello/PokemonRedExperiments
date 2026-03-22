import sys
from datetime import UTC, datetime
from os.path import exists
from pathlib import Path
from typing import Any, cast

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv
from wandb.integration.sb3 import WandbCallback

import wandb
from red_gym_env_v2 import EnvConfig, RedGymEnv
from stream_agent_wrapper import StreamWrapper
from tensorboard_callback import TensorboardCallback


def make_env(rank: int, env_conf: EnvConfig, seed: int = 0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        env = StreamWrapper(
            RedGymEnv(env_conf),
            stream_metadata={  # All of this is part is optional
                "user": "marcus",  # choose your own username
                "env_id": rank,  # environment identifier
                "color": "#447799",  # choose your color :)
                "extra": "extra-text",  # any extra text you put here will be displayed
                "sprite_id": 10,
            },
        )
        _ = env.reset(seed=(seed + rank))
        return env

    set_random_seed(seed)
    return _init


if __name__ == "__main__":
    use_wandb_logging = True
    ep_length = 2048 * 80
    sess_id = f"poke-v2-run-{datetime.now(UTC).strftime('%Y%m%d_%H%M')}"
    sess_path = Path("runs")

    env_config: EnvConfig = {
        "headless": True,
        "save_final_state": False,
        "early_stop": False,
        "action_freq": 24,
        "init_state": "../init.state",
        "max_steps": ep_length,
        "print_rewards": True,
        "save_video": False,
        "fast_video": False,
        "session_path": sess_path,
        "gb_path": "../PokemonRed.gb",
        "debug": False,
        "reward_scale": 0.5,
        "explore_weight": 0.25,
    }

    print(env_config)

    num_cpu = 64  # Also sets the number of episodes per training iteration
    env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])

    checkpoint_callback = CheckpointCallback(save_freq=ep_length // 2, save_path=str(sess_path), name_prefix="poke")

    callbacks: list[BaseCallback] = [checkpoint_callback, TensorboardCallback(sess_path)]

    run = None
    if use_wandb_logging:
        run = wandb.init(
            project="pokemon-train",
            id=sess_id,
            name=f"v2-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}",
            config=cast(dict[str, Any], cast(object, env_config)),  # pyright is fussy here, double cast
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )
        callbacks.append(WandbCallback())

    # env_checker.check_env(env)

    # put a checkpoint here you want to start from
    if sys.stdin.isatty():
        file_name = ""
    else:
        file_name = sys.stdin.read().strip()  # "runs/poke_26214400_steps"

    train_steps_batch = ep_length // 64

    if exists(file_name + ".zip"):
        print("\nloading checkpoint")
        model = PPO.load(file_name, env=env)
        model.n_steps = train_steps_batch
        model.n_envs = num_cpu
        model.rollout_buffer.buffer_size = train_steps_batch
        model.rollout_buffer.n_envs = num_cpu
        model.rollout_buffer.reset()
    else:
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            n_steps=train_steps_batch,
            batch_size=512,
            n_epochs=1,
            gamma=0.997,
            ent_coef=0.01,
            tensorboard_log=str(sess_path),
        )

    print(model.policy)

    _ = model.learn(
        total_timesteps=(ep_length) * num_cpu * 10000,
        callback=CallbackList(callbacks),
        tb_log_name="poke_ppo",
    )

    if run is not None:
        run.finish()
