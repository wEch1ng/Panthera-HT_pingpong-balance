"""Image-only ball feedback; joint encoders/history complete the policy input."""
import numpy as np
from perception.detector import detect
from perception.estimator import EstRunner, load_est
from sim.env import FdVel
from sim.policies import controller_action


class VisionController:
    def __init__(self, controller, estimator_path):
        if controller.train_args.get('recovery_task'):
            raise ValueError('Recovery policy needs RGB-D observations, not the old 2D estimator')
        self.controller = controller
        self.estimator = EstRunner(load_est(estimator_path))
        self.fd = FdVel()

    def reset(self):
        self.estimator.reset()
        self.fd.reset()

    def action(self, env, rgb):
        detection = detect(rgb)
        position = self.estimator.update(detection)[:2]
        ball = self.fd.update(position)
        # Deliberately do not read ball qpos/qvel or env.controller_obs().
        q = (env.d.qpos[env.q_adr] - env.home_q) / 0.2
        qd = env.d.qvel[env.v_adr] * 0.1
        obs = np.concatenate([ball, q, qd, env._hist.ravel()]).astype(np.float32)
        return controller_action(self.controller, obs), detection, ball
