"""Load the PPO controller and query its action. Shared by the viewer, the evals and hardware/run_real.py."""
from __future__ import annotations

import numpy as np
import torch

from sim.ppo import ActorCritic
from sim.robot import ROBOT_ID, N_ARM, BASE_OBS


def load_controller(path) -> ActorCritic:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    obs_dim = next(v for k, v in ck["model"].items() if k.startswith("actor") and k.endswith("weight")).shape[1]
    act_dim = ck["model"]["log_std"].numel()
    extra = 0
    if ck.get('args', {}).get('recovery_task'):
        from sim.recovery_env import EXTRA_OBS, OBS_ID
        if ck.get('args', {}).get('observation_id') != OBS_ID:
            raise ValueError('Recovery policy observation version mismatch')
        extra = EXTRA_OBS
    if act_dim != N_ARM or ck.get("robot_id") != ROBOT_ID:
        raise ValueError("This checkpoint is not a Panthera six-joint policy. Train with sim.train_controller; SO-101 weights cannot be reused.")
    if obs_dim < BASE_OBS + N_ARM + extra or (obs_dim - BASE_OBS - extra) % N_ARM:
        raise ValueError(f"Invalid Panthera observation dimension: {obs_dim}")
    ac = ActorCritic(obs_dim, act_dim)
    ac.load_state_dict(ck["model"])
    ac.act_hist = (obs_dim - BASE_OBS - extra) // N_ARM  # extra recovery features are appended
    ac.train_args = ck.get("args", {})  # the env settings it was trained with (latency, servo randomization, ball noise, grip)
    return ac.eval()


@torch.no_grad()
def controller_action(ac: ActorCritic, tobs: np.ndarray) -> np.ndarray:
    """The controller's mean action (no exploration noise), clipped to the env's range."""
    return np.clip(ac.dist(torch.as_tensor(tobs[None], dtype=torch.float32)).mean[0].numpy(), -1, 1)
