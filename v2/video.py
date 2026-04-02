import glob
import os
import time
from datetime import UTC, datetime
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

from red_gym_env_v2 import EnvConfig, RedGymEnv


def make_env(rank: int, env_conf: EnvConfig, seed: int = 0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        env = RedGymEnv(env_conf)
        return env

    set_random_seed(seed)
    return _init


def get_most_recent_zip_with_age(folder_path: Path) -> tuple[str | None, float | None]:
    # Get all zip files in the folder
    zip_files = glob.glob(os.path.join(folder_path, "*.zip"))

    if not zip_files:
        return None, None  # Return None if no zip files are found

    # Find the most recently modified zip file
    most_recent_zip = max(zip_files, key=os.path.getmtime)

    # Calculate how old the file is in hours
    current_time = time.time()
    modification_time = os.path.getmtime(most_recent_zip)
    age_in_hours = (current_time - modification_time) / 3600  # Convert seconds to hours

    return most_recent_zip, age_in_hours


if __name__ == "__main__":
    sess_id = f"poke-v2-video-{datetime.now(UTC).strftime('%Y%m%d_%H%M')}"
    sess_path = Path(f"video/{sess_id}")
    ep_length = 2**23

    env_config: EnvConfig = {
        "headless": True,
        "save_final_state": False,
        "early_stop": False,
        "action_freq": 24,
        "init_state": "../init.state",
        "max_steps": ep_length,
        "print_rewards": True,
        "save_video": True,
        "fast_video": True,
        "session_path": sess_path,
        "gb_path": "../PokemonRed.gb",
        "debug": False,
        "sim_frame_dist": 2_000_000.0,
        "extra_buttons": False,
    }

    num_cpu = 1  # Also sets the number of episodes per training iteration
    env = make_env(0, env_config)()

    most_recent_checkpoint, time_since = get_most_recent_zip_with_age(Path("./runs/checkpoints"))
    if most_recent_checkpoint is None:
        raise FileNotFoundError("No checkpoint found in ./runs/checkpoints")
    print(f"using checkpoint: {most_recent_checkpoint}, which is {time_since} hours old")
    model = PPO.load(most_recent_checkpoint, env=env, custom_objects={"lr_schedule": 0, "clip_range": 0})

    obs, info = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=False)
        obs, rewards, terminated, truncated, info = env.step(int(action))

        env.render()
        if truncated:
            break
    env.close()
