"""D405 RGB-D recovery task: catch, recenter, restore height, minimize work.

Actor observations come from rendered RGB/depth and joint kinematics only.
Simulator truth is used exclusively for reward, resets and evaluation metrics.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
from sim.env import BalanceEnv  # selects the headless GL backend before MuJoCo import
import mujoco
from mujoco.rendering.classic.renderer import Renderer
import numpy as np
from gymnasium import spaces
from sim.robot import ARM_JOINTS, HOME_Q, BASE_OBS, N_ARM, SOURCE_URDF
from perception.rgbd_ball import RGBDBallTracker, locate_ball

OBS_ID = 'd405-rgbd-recovery-v1'
EXTRA_OBS = 8


class RecoveryEnv(BalanceEnv):
    def __init__(self, target_height=.527232155, target_u=0., target_w=0.,
                 energy_weight=.02, recovery_level=1., **kwargs):
        kwargs.update(obs_mode='controller', light_model=False, recovery_mode=True,
                      ball_noise=0., home_penalty=0., cam_rand=0.)
        kwargs.setdefault('max_steps', 600)
        super().__init__(**kwargs)
        # Some MuJoCo builds suppress renderer ImportErrors at package import.
        # Import explicitly above so worker failures include the real cause.
        if not hasattr(mujoco, 'Renderer'):
            mujoco.Renderer = Renderer
        self.target_height = float(target_height)
        self.target_uv = np.array([target_u, target_w])
        if not .25 <= self.target_height <= .65 or np.linalg.norm(self.target_uv) > .04:
            raise ValueError('Target height must be 0.25..0.65 m; center offset <= 4 cm')
        if energy_weight < 0:
            raise ValueError('energy_weight must be nonnegative')
        self.energy_weight = energy_weight
        self.set_curriculum(recovery_level)
        root = ET.parse(SOURCE_URDF).getroot()
        self.q_speed_limits = np.array([float(root.find(f"joint[@name='{j}']/limit").get('velocity')) for j in ARM_JOINTS])
        self.tracker = RGBDBallTracker()
        self.observation_space = spaces.Box(-np.inf, np.inf,
            (BASE_OBS + N_ARM*self.act_hist + EXTRA_OBS,), np.float32)
        self._sensor_time = None
        self.last_det = np.zeros(4)
        self.episode_energy_j = 0.
        self.stable_steps = 0
        self.min_blade_height = 1.
        self._kick_at = -1
        self._kick = np.zeros(3)

    def set_curriculum(self, level):
        self.recovery_level = float(np.clip(level, 0., 1.))

    def reset(self, *, seed=None, options=None):
        self.tracker.reset()
        self._sensor_time = None
        self.episode_energy_j = 0.
        self.stable_steps = 0
        self._kick_at = -1
        super().reset(seed=seed, options=options)
        r, level = self.rng, self.recovery_level
        # Mix nominal balancing, fast rolling, drops and initially-low paddle poses.
        self.scenario = str(r.choice(['balance', 'rolling', 'drop', 'low_return'],
                                    p=[.25, .30, .30, .15]))
        if options and 'scenario' in options:
            if options['scenario'] not in ('balance','rolling','drop','low_return'):
                raise ValueError('Unknown recovery scenario')
            self.scenario = options['scenario']
        if self.scenario == 'low_return':
            # Small IK solve: lower the blade while preserving its home orientation.
            self.d.qpos[self.q_adr] = HOME_Q
            mujoco.mj_forward(self.m, self.d)
            desired = self.d.geom_xpos[self.blade_geom].copy()
            desired[2] -= r.uniform(.01, .02+.08*level)
            jp, jr = np.zeros((3,self.m.nv)), np.zeros((3,self.m.nv))
            for _ in range(25):
                c, n, _, _ = self.blade_frame()
                mujoco.mj_jacGeom(self.m, self.d, jp, jr, self.blade_geom)
                jac = np.vstack((jp[:,self.v_adr], jr[:,self.v_adr]))
                err = np.r_[desired-c, np.cross(n, [0,0,1])]
                dq = jac.T @ np.linalg.solve(jac@jac.T + .002*np.eye(6), err)
                self.d.qpos[self.q_adr] = np.clip(self.d.qpos[self.q_adr]+np.clip(dq,-.08,.08), self.q_lo,self.q_hi)
                mujoco.mj_forward(self.m,self.d)
            self._q_target = self.d.qpos[self.q_adr].copy()
            self.d.ctrl[self.act_ids] = self._q_target
        c, n, u, w = self.blade_frame()
        angle = r.uniform(0, 2*np.pi)
        offset = r.uniform(0., .015 + .045*level)
        gap = r.uniform(.01, .02+.10*level) if self.scenario == 'drop' else .003
        self.d.qpos[self.ball_qadr:self.ball_qadr+3] = (
            c + n*(self.ball_radius+self.m.geom_size[self.blade_geom,1]+gap)
            + u*(self.target_uv[0]+offset*np.cos(angle)) + w*(self.target_uv[1]+offset*np.sin(angle)))
        self.d.qvel[self.ball_vadr:self.ball_vadr+6] = 0.
        if self.scenario in ('rolling', 'drop'):
            magnitude = r.uniform(.03, .06+.29*level)
            self.d.qvel[self.ball_vadr:self.ball_vadr+3] = magnitude*(np.cos(angle)*u+np.sin(angle)*w)
        if self.scenario == 'drop':
            self.d.qvel[self.ball_vadr+2] -= r.uniform(0., .10+.35*level)
        if r.random() < .5:
            self._kick_at = int(r.integers(90, 240))
            angle = r.uniform(0,2*np.pi)
            self._kick = (.03+.22*level)*np.array([np.cos(angle),np.sin(angle),0.])
        mujoco.mj_forward(self.m,self.d)
        self.min_blade_height = float(c[2])
        self.tracker.reset()
        self._sensor_time = None
        return self._obs(), self._info()

    def controller_obs(self):
        if self._sensor_time != self.d.time:
            self.latest_rgb = self._render()
            self._renderer.enable_depth_rendering()
            self._renderer.update_scene(self.d, camera=self.cam_id)
            depth = self._renderer.render().copy()
            self._renderer.disable_depth_rendering()
            local, self.last_det = locate_ball(self.latest_rgb, depth, self.m.cam_fovy[self.cam_id], self.ball_radius)
            point = None
            if local is not None:
                point = self.d.cam_xpos[self.cam_id] + self.d.cam_xmat[self.cam_id].reshape(3,3) @ local
                if self.domain_rand:
                    point += self.rng.normal(0,.0015,3)
                    if self.rng.random() < .02:
                        point = None
            self._tracked = self.tracker.update(point)
            self._sensor_time = float(self.d.time)
        point, velocity, visible, age = self._tracked
        c, n, u, w = self.blade_frame()
        # An unseen ball is explicitly marked, never filled from true ball state.
        rel = np.zeros(3) if point is None else point-c
        ball = np.array([rel@u/self.blade_radius, rel@w/self.blade_radius,
                         velocity@u/.5, velocity@w/.5])
        jp, jr = np.zeros((3,self.m.nv)), np.zeros((3,self.m.nv))
        mujoco.mj_jacGeom(self.m,self.d,jp,jr,self.blade_geom)
        blade_vz = float(jp[2,self.v_adr] @ self.d.qvel[self.v_adr])
        extra = [rel@n/.1, velocity@n/.5, visible, min(age/.3,5.),
                 (c[2]-self.target_height)/.1, *self.target_uv/self.blade_radius, blade_vz/.5]
        return np.r_[ball, (self.d.qpos[self.q_adr]-self.home_q)/.2,
                     self.d.qvel[self.v_adr]*.1, self._hist.ravel(), extra].astype(np.float32)

    def step(self, action):
        if self.t == self._kick_at:
            self.d.qvel[self.ball_vadr:self.ball_vadr+3] += self._kick
        previous_action = self._prev_action.copy()
        obs, _, term, trunc, info = super().step(action)
        c, n, u, w = self.blade_frame()
        bu, bw, h, speed = self.ball_in_blade()
        dist = float(np.linalg.norm(np.array([bu,bw])-self.target_uv))
        gap = h - self.ball_radius - self.m.geom_size[self.blade_geom,1]
        contact = any({con.geom1,con.geom2} == {self.ball_geom,self.blade_geom}
                      and con.dist <= .001 for con in self.d.contact)
        height_error = float(c[2]-self.target_height)
        caught = contact and abs(gap)<.02
        settled = caught and dist < .02 and speed < .08
        goal = settled and abs(height_error)<.02
        self.stable_steps = self.stable_steps+1 if goal else 0
        self.episode_energy_j += self.step_energy_j
        self.min_blade_height = min(self.min_blade_height,float(c[2]))
        # During danger, guide the blade under a short-horizon ballistic intercept.
        # Once caught, the height goal gradually regains weight.
        vel = self.d.qvel[self.ball_vadr:self.ball_vadr+3]
        ball_pos = self.d.xpos[self.ball_body]
        aim = ball_pos + .10*vel + np.array([0,0,-.5*9.81*.10**2])
        aim -= n*(self.ball_radius+self.m.geom_size[self.blade_geom,1])
        aim -= u*self.target_uv[0]+w*self.target_uv[1]
        aim[2] = np.clip(aim[2],.18,self.target_height)
        near = np.exp(-(dist/.06)**2)
        on_face = np.exp(-(gap/.035)**2)
        stable_gate = near*on_face*np.exp(-(speed/.25)**2)
        power_w = self.step_energy_j/(self.n_substeps*self.m.opt.timestep)
        torque_ratio = self.d.actuator_force[self.act_ids]/np.maximum(self.m.actuator_forcerange[self.act_ids,1],1.)
        reward = (2*near*on_face + .5*np.exp(-np.sum(((c-aim)/.12)**2))
                  + stable_gate*np.exp(-(height_error/.03)**2)
                  - .3*stable_gate*min((height_error/.1)**2,4.)
                  - .05*min(speed,3.) - self.energy_weight*power_w
                  - .01*float(np.sum((np.clip(action,-1,1)-previous_action)**2))
                  - .002*float(np.mean(torque_ratio**2)))
        overspeed = bool(np.any(self.step_peak_speed > self.q_speed_limits))
        limit = bool(np.any(self.d.qpos[self.q_adr] < self.q_lo-.02) or
                     np.any(self.d.qpos[self.q_adr] > self.q_hi+.02))
        low_blade = c[2] < .15
        term = bool(term or overspeed or limit or low_blade)
        if overspeed or limit or low_blade:
            info['termination_reason'] = 'speed_limit' if overspeed else 'joint_limit' if limit else 'low_blade'
        if term:
            reward -= 10.
        info.update(is_success=bool(trunc and not term and self.stable_steps>=30),
                    power_w=power_w, energy_j=self.episode_energy_j,
                    height_error_m=height_error, min_blade_height_m=self.min_blade_height,
                    stable_steps=self.stable_steps, scenario=self.scenario,
                    recovery_level=self.recovery_level, caught=bool(caught))
        return obs,float(reward),term,trunc,info
