"""Watch the trained PPO controller balance the ball in the live MuJoCo viewer.
Usage: uv run python -m sim.view_controller [--controller outputs/controller/panthera/last.pt] [--rand] [--latency N] [--seed N] [--nudge SECONDS]
  --rand      domain randomization on, with the settings the controller was trained with (latency, soft servos, noisy ball state)
  --latency   force a control latency of N extra steps (33 ms each)
  --nudge S   kick the ball sideways every S seconds (you can also Ctrl+right-drag the ball with the mouse)
The policy only gets the controller observation (ball state in the blade frame + joints + recent actions); the actions are the policy mean."""
import argparse
import time
from threading import Event
from pathlib import Path

import mujoco.viewer
import numpy as np
import cv2
import torch
torch.set_num_threads(1)
from perception.vision_control import VisionController

from sim.env import BalanceEnv
from sim.policies import controller_action, load_controller

p = argparse.ArgumentParser()
p.add_argument("--controller", default=None, help="Panthera checkpoint; omit for pose preview"); p.add_argument("--rand", action="store_true")
p.add_argument("--latency", type=int, default=None); p.add_argument("--seed", type=int, default=0); p.add_argument("--nudge", type=float, default=0.0)
p.add_argument("--estimator", help="D405 estimator; enable image-only ball feedback")
p.add_argument("--camera", action="store_true", help="Show wrist camera image overlay")
p.add_argument("--grip-rot", type=float, default=0.0)
p.add_argument("--strict-episodes", action="store_true", help="Use the original 10-second episodes and edge cutoff")
a = p.parse_args()

if a.estimator and not a.controller:
    p.error("--estimator requires --controller")
ac = load_controller(a.controller) if a.controller else None
rgbd = bool(ac is not None and ac.train_args.get('recovery_task'))
if rgbd and a.estimator:
    p.error('RGB-D recovery policy uses its own 3D tracker; omit --estimator')
ta = ac.train_args if a.rand and ac is not None else {}
vision = VisionController(ac, a.estimator) if a.estimator else None
if rgbd:
    from sim.recovery_env import RecoveryEnv
    settings = ac.train_args
    env = RecoveryEnv(act_hist=ac.act_hist, domain_rand=a.rand,
                      target_height=settings['target_height'], target_u=settings['target_u'], target_w=settings['target_w'],
                      energy_weight=settings['energy_weight'], recovery_level=settings['recovery_level'],
                      max_latency=ta.get('max_latency', 0), max_steps=600 if a.strict_episodes else None)
else:
    env = BalanceEnv(obs_mode="image" if (a.camera or vision) else "controller", domain_rand=a.rand, light_model=False, grip_rot_deg=a.grip_rot, act_hist=ac.act_hist if ac is not None else 1,  # full visual model so the arm and paddle are drawn
                 max_steps=300 if a.strict_episodes else None, recovery_mode=not a.strict_episodes,
                 max_latency=ta.get("max_latency", 2), servo_rand=ta.get("servo_rand", False), ball_noise=ta.get("ball_noise", 0.0))
seed = a.seed
print("Feedback: D405 rendered RGB-D -> 3D tracker" if rgbd else "Feedback: D405 RGB -> detector -> estimator -> FdVel" if vision else "Feedback: simulator state / pose preview", flush=True)
obs, _ = env.reset(seed=seed)
if a.latency is not None:
    env._action_delay = a.latency; env._delay_buf = [np.zeros(6)] * a.latency
print(f"mode {a.controller or 'POSE PREVIEW (no trained controller)'}; randomization {'on' if a.rand else 'off'}; latency {env._action_delay}")
rng = np.random.default_rng(seed)
reset_requested = Event()
def on_key(key):
    if key in (ord('R'), ord('r')):
        reset_requested.set()

print("Episode mode: " + (f"timed {env.max_steps/30:g} seconds" if a.strict_episodes else "continuous recovery; floor/distance/physics limits apply") + "; R = reset", flush=True)
with mujoco.viewer.launch_passive(env.m, env.d, key_callback=on_key) as v:
    v.opt.geomgroup[3] = 0
    v.opt.flags[mujoco.mjtVisFlag.mjVIS_ISLAND] = False
    v.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    v.cam.lookat[:] = [0.25, 0.0, 0.25]; v.cam.distance = 1.3; v.cam.azimuth = 135; v.cam.elevation = -20
    last_nudge = time.time(); episodes = successes = 0
    while v.is_running():
        t0 = time.time()
        if vision:
            action, det, ball_est = vision.action(env, obs["image"])
        else:
            action = controller_action(ac, env.controller_obs()) if ac is not None else np.zeros(6)
            det, ball_est = None, None
        if rgbd:
            det, ball_est = env.last_det, obs[:4]
        if a.camera or vision or rgbd:
            img = cv2.resize(env.latest_rgb if rgbd else obs["image"], (320, 240), interpolation=cv2.INTER_NEAREST)
            if det is not None and det[3]:
                center = (int((det[0] + 1) * 160), int((det[1] + 1) * 120))
                cv2.circle(img, center, max(2, int(det[2] * 25)), (0,255,0), 2)
            label = "D405 RGB-D RECOVERY" if rgbd else "D405 VISION CONTROL" if vision else "D405 CAMERA (state control)"
            cv2.putText(img, label, (6,18), cv2.FONT_HERSHEY_SIMPLEX, .43, (255,255,255), 1)
            cv2.putText(img, f"{'STRICT' if a.strict_episodes else 'RECOVERY'} {env.t/30:.1f}s | R: reset",
                        (6,36), cv2.FONT_HERSHEY_SIMPLEX, .4, (255,255,255), 1)
            if vision or rgbd:
                cv2.putText(img, f"visible={int(det[3])}  u/w={ball_est[0]*80:.1f}/{ball_est[1]*80:.1f} mm",
                            (6,232), cv2.FONT_HERSHEY_SIMPLEX, .4, (255,255,255), 1)
            vp = v.viewport
            if vp is not None:
                v.set_images((mujoco.MjrRect(vp.left + max(0,vp.width-330), vp.bottom+10, 320,240), img))
        obs, r, te, tr, info = env.step(action)
        if a.nudge and time.time() - last_nudge > a.nudge:
            ang = rng.uniform(0, 2 * np.pi); env.d.qvel[env.ball_vadr:env.ball_vadr + 2] += 0.25 * np.array([np.cos(ang), np.sin(ang)]); last_nudge = time.time()
        # Keep CAD materials instead of debug island colors/transparency.
        v.opt.flags[mujoco.mjtVisFlag.mjVIS_ISLAND] = False
        v.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
        if v.user_scn is not None:
            v.user_scn.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = False
            v.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
            v.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = True
        v.sync()
        manual_reset = reset_requested.is_set()
        if not a.strict_episodes and env.t % 900 == 0 and not te:
            print(f"continuous run: {env.t/30:.0f}s, ball offset {info['dist']*1000:.1f} mm", flush=True)
        if te or tr or manual_reset:
            episodes += 1; successes += int(info["is_success"])
            result = 'manual_reset' if manual_reset else info['termination_reason'] or 'time_limit'
            print(f"run {episodes}: {result} (duration {env.t/30:.1f}s, final dist {info['dist']*1000:.1f} mm)", flush=True)
            reset_requested.clear()
            seed += 1; obs, _ = env.reset(seed=seed)
            if vision:
                vision.reset()
            if a.latency is not None:
                env._action_delay = a.latency; env._delay_buf = [np.zeros(6)] * a.latency
        time.sleep(max(0, env.n_substeps * env.m.opt.timestep - (time.time() - t0)))

env.close()
