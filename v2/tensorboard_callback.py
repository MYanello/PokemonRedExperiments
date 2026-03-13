import json
import os
from pathlib import Path
from typing import Any, cast, override

import numpy as np
from einops import rearrange, reduce
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Image
from torch.utils.tensorboard import SummaryWriter


def merge_dicts(dicts: list[dict[str, int | float | list[int]]]) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    sum_dict = {}
    count_dict = {}
    distrib_dict = {}

    for d in dicts:
        for k, v in d.items():
            if isinstance(v, (int, float)):
                sum_dict[k] = sum_dict.get(k, 0) + v
                count_dict[k] = count_dict.get(k, 0) + 1
                distrib_dict.setdefault(k, []).append(v)

    mean_dict = {}
    for k in sum_dict:
        mean_dict[k] = sum_dict[k] / count_dict[k]
        distrib_dict[k] = np.array(cast(list[float], distrib_dict[k]))

    return mean_dict, distrib_dict


class TensorboardCallback(BaseCallback):
    def __init__(self, log_dir: Path, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir: Path = log_dir
        self.writer: SummaryWriter | None = None

    @override
    def _on_training_start(self) -> None:
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=os.path.join(self.log_dir, "histogram"))

    @override
    def _on_step(self) -> bool:
        if self.training_env.env_method("check_if_done", indices=[0])[0]:
            all_infos = cast(list[list[dict[str, Any]]], self.training_env.get_attr("agent_stats"))
            all_final_infos = [stats[-1] for stats in all_infos]
            mean_infos, distributions = merge_dicts(all_final_infos)
            # TODO log distributions, and total return
            for key, val in mean_infos.items():
                self.logger.record(f"env_stats/{key}", val)

            if self.writer:
                for key, distrib in distributions.items():
                    self.writer.add_histogram(f"env_stats_distribs/{key}", distrib, self.n_calls)
                    self.logger.record(f"env_stats_max/{key}", max(distrib))

            # images = self.training_env.get_attr("recent_screens")
            # images_row = rearrange(np.array(images), "(r f) h w c -> (r c h) (f w)", r=2)
            # self.logger.record("trajectory/image", Image(images_row, "HW"), exclude=("stdout", "log", "json", "csv"))

            explore_map = np.array(self.training_env.get_attr("explore_map"))
            map_sum = reduce(explore_map, "f h w -> h w", "max")
            self.logger.record("trajectory/explore_sum", Image(map_sum, "HW"), exclude=("stdout", "log", "json", "csv"))

            map_row = rearrange(explore_map, "(r f) h w -> (r h) (f w)", r=2)
            self.logger.record("trajectory/explore_map", Image(map_row, "HW"), exclude=("stdout", "log", "json", "csv"))

            list_of_flag_dicts = cast(list[dict[str, str]], self.training_env.get_attr("current_event_flags_set"))
            merged_flags = {k: v for d in list_of_flag_dicts for k, v in d.items()}
            self.logger.record("trajectory/all_flags", json.dumps(merged_flags))

        return True

    @override
    def _on_training_end(self) -> None:
        if self.writer:
            self.writer.close()
