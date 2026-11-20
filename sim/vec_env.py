"""Parallel BalanceEnvs: n_workers processes, each stepping envs_per_worker envs per pipe message (an env step is only ~0.25 ms,
so one message per env would cost more than the physics).

step() auto-resets finished envs and returns (obs, rew, terminated, truncated, final_obs, episodes):
final_obs = the real last obs of a finished episode (for value bootstrapping at the time limit), episodes = stats of finished episodes.
"""
from __future__ import annotations

import multiprocessing as mp

import numpy as np

from sim.ram import check_ram
from sim.robot import N_ARM, BASE_OBS


def _worker(conn, env_kwargs, seeds):
    from sim.env import BalanceEnv
    options = dict(env_kwargs)
    recovery = options.pop('recovery_task', False)
    if recovery:
        from sim.recovery_env import RecoveryEnv
        env_type = RecoveryEnv
    else:
        env_type = BalanceEnv
    envs = [env_type(obs_mode="controller", **options) for _ in seeds]
    ret = np.zeros(len(envs))
    length = np.zeros(len(envs), int)
    for e, s in zip(envs, seeds):
        e.reset(seed=s)
    while True:
        cmd, data = conn.recv()
        if cmd == "reset":
            conn.send(np.stack([e.reset()[0] for e in envs]))
        elif cmd == "step":
            obs, rew, term, trunc, final, eps = [], [], [], [], [], []
            for i, (e, a) in enumerate(zip(envs, data)):
                o, r, te, tr, info = e.step(a)
                ret[i] += r
                length[i] += 1
                fo = np.zeros_like(o)
                if te or tr:
                    fo = o
                    eps.append(dict(ret=ret[i], length=int(length[i]), success=bool(info["is_success"]), latency=e._action_delay,
                                    **{k: info[k] for k in ('energy_j','height_error_m','min_blade_height_m') if k in info}))
                    ret[i], length[i] = 0.0, 0
                    o, _ = e.reset()
                obs.append(o); rew.append(r); term.append(te); trunc.append(tr); final.append(fo)
            conn.send((np.stack(obs), np.array(rew, np.float32), np.array(term), np.array(trunc), np.stack(final), eps))
        elif cmd == "close":
            for e in envs:
                e.close()
            conn.close()
            return
        elif cmd == "curriculum":
            for e in envs:
                e.set_curriculum(data)
            conn.send(True)


class VecBalanceEnv:
    def __init__(self, n_workers: int = 6, envs_per_worker: int = 4, seed: int = 0, **env_kwargs):
        # ~135 MB per env (measured): refuse to start rather than run out of RAM
        check_ram(1.5 * 135 * n_workers * envs_per_worker, "Use fewer envs (--workers / --envs-per-worker) and raise --steps to keep the batch size, see the README.")
        ctx = mp.get_context("spawn")  # fork is unsafe with torch/CUDA in the parent
        self.n_envs = n_workers * envs_per_worker
        self.conns, self.procs = [], []
        for w in range(n_workers):
            parent, child = ctx.Pipe()
            seeds = [seed + w * envs_per_worker + k for k in range(envs_per_worker)]
            p = ctx.Process(target=_worker, args=(child, env_kwargs, seeds), daemon=True)
            p.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(p)
        self.per = envs_per_worker
        self.obs_dim, self.act_dim = BASE_OBS + N_ARM * env_kwargs.get("act_hist", 1), N_ARM
        if env_kwargs.get('recovery_task'):
            from sim.recovery_env import EXTRA_OBS
            self.obs_dim += EXTRA_OBS

    def set_curriculum(self, level):
        for c in self.conns:
            c.send(('curriculum', level))
        for c in self.conns:
            c.recv()

    def reset(self):
        for c in self.conns:
            c.send(("reset", None))
        return np.concatenate([c.recv() for c in self.conns])

    def step(self, actions):
        actions = np.asarray(actions, np.float32).reshape(self.n_envs, self.act_dim)
        for i, c in enumerate(self.conns):  # send all first, so workers run in parallel
            c.send(("step", actions[i * self.per : (i + 1) * self.per]))
        out = [c.recv() for c in self.conns]
        obs, rew, term, trunc, final = (np.concatenate([o[k] for o in out]) for k in range(5))
        episodes = [ep for o in out for ep in o[5]]
        return obs, rew, term, trunc, final, episodes

    def close(self):
        for c in self.conns:
            try:
                c.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=5)
