import asyncio
import json
import logging
import os
from asyncio.events import AbstractEventLoop
from typing import override

import gymnasium as gym
import numpy as np
import websockets
from pyboy import PyBoy
from websockets.asyncio.client import ClientConnection

from red_gym_env_v2 import RedGymEnv

X_POS_ADDRESS, Y_POS_ADDRESS = 0xD362, 0xD361
MAP_N_ADDRESS = 0xD35E

logging.basicConfig(
    filename=os.path.join(os.path.dirname(__file__), "logs/stream_wrapper.log"),
    level=logging.DEBUG,
    format="%(asctime)s [pid %(process)d] %(levelname)s: %(message)s",
)


class StreamWrapper(gym.Wrapper[dict[str, np.ndarray], int, dict[str, np.ndarray], int]):
    def __init__(self, env: RedGymEnv, stream_metadata: dict[str, str | int] | None = None):
        super().__init__(env)
        self.ws_address: str = "ws://pokerl-map-viz:3344/broadcast"
        self.stream_metadata: dict[str, str | int] = stream_metadata or {}
        self.loop: AbstractEventLoop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.websocket: ClientConnection | None = None
        self.loop.run_until_complete(self.establish_wc_connection())
        self.upload_interval: int = 300
        self.steam_step_counter: int = 0
        self.env: RedGymEnv = env
        self.coord_list: list = []
        self.emulator: PyBoy = env.pyboy

    @override
    def step(self, action: int):
        x_pos = self.emulator.memory[X_POS_ADDRESS]
        y_pos = self.emulator.memory[Y_POS_ADDRESS]
        map_n = self.emulator.memory[MAP_N_ADDRESS]
        self.coord_list.append([x_pos, y_pos, map_n])

        if self.steam_step_counter >= self.upload_interval:
            self.stream_metadata["extra"] = f"coords: {len(getattr(self.env, 'seen_coords', {}))}"
            self.loop.run_until_complete(self.broadcast_ws_message(json.dumps({"metadata": self.stream_metadata, "coords": self.coord_list})))
            self.steam_step_counter = 0
            self.coord_list = []

        self.steam_step_counter += 1

        return self.env.step(action)

    async def broadcast_ws_message(self, message: str | bytes) -> None:
        if self.websocket is None:
            await self.establish_wc_connection()
        if self.websocket is not None:
            try:
                await self.websocket.send(message)
            except websockets.exceptions.WebSocketException:
                self.websocket = None

    async def establish_wc_connection(self) -> None:
        try:
            self.websocket = await websockets.connect(self.ws_address)
            logging.info("Connection succeeded")
        except Exception as e:
            logging.error(f"Connection failed: {e}")
            self.websocket = None
