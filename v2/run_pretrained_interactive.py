import glob
import os
import time
import uuid
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

from red_gym_env_v2 import EnvConfig, RedGymEnv
from stream_agent_wrapper import StreamWrapper


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
        env = StreamWrapper(
            env,
            stream_metadata={
                "user": "marcus",
                "env_id": rank,
                "color": "#0033ff",
                "extra": "",
            },
        )
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
    sess_path = Path(f"session_{str(uuid.uuid4())[:8]}")
    ep_length = 2**23

    env_config: EnvConfig = {
        "headless": False,
        "save_final_state": True,
        "early_stop": False,
        "action_freq": 24,
        "init_state": "../init.state",
        "max_steps": ep_length,
        "print_rewards": True,
        "save_video": False,
        "fast_video": True,
        "session_path": sess_path,
        "gb_path": "../PokemonRed.gb",
        "debug": False,
        "sim_frame_dist": 2_000_000.0,
        "extra_buttons": False,
    }

    num_cpu = 1  # 64 #46  # Also sets the number of episodes per training iteration
    env = make_env(0, env_config)()  # SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])

    # env_checker.check_env(env)
    most_recent_checkpoint, time_since = get_most_recent_zip_with_age(Path("runs"))
    if most_recent_checkpoint is None:
        raise FileNotFoundError("No checkpoint found in runs/")
    file_name = most_recent_checkpoint
    print(f"using checkpoint: {file_name}, which is {time_since} hours old")
    # could optionally manually specify a checkpoint here
    # file_name = "runs/poke_41943040_steps.zip"
    print("\nloading checkpoint")
    model = PPO.load(file_name, env=env, custom_objects={"lr_schedule": 0, "clip_range": 0})

    # keyboard.on_press_key("M", toggle_agent)
    obs, info = env.reset()
    while True:
        try:
            with open("agent_enabled.txt", "r") as f:
                agent_enabled = f.readlines()[0].startswith("yes")
        except Exception:
            agent_enabled = False
        if agent_enabled:
            action, _states = model.predict(obs, deterministic=False)
            obs, rewards, terminated, truncated, info = env.step(int(action))
        else:
            env.emulator.tick(1, True)
            obs = env.env.get_obs()
            truncated = env.env.step_count >= env.env.max_steps - 1
        env.render()
        if truncated:
            break
    env.close()
