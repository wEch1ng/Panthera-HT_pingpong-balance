"""Integration checks for the URDF adaptation. Run: python -m unittest discover -s tests -v."""
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
import mujoco
import numpy as np
from sim.env import BalanceEnv
from sim.robot import ARM_JOINTS, HOME_Q, SOURCE_URDF
from sim.policies import load_controller


class PantheraTests(unittest.TestCase):
    def test_recovery_allows_edge_and_airborne_excursions(self):
        for recovery in (False, True):
            e = BalanceEnv(recovery_mode=recovery, max_steps=None)
            self.addCleanup(e.close)
            for offset, height in ((.09, .03), (0., .12), (.09, -.02)):
                e.reset(seed=0)
                c, n, u, _ = e.blade_frame()
                e.d.qpos[e.ball_qadr:e.ball_qadr+3] = c + offset*u + height*n
                e.d.qvel[e.ball_vadr:e.ball_vadr+6] = 0
                mujoco.mj_forward(e.m, e.d)
                _, _, term, trunc, _ = e.step(np.zeros(6))
                self.assertEqual(term, not recovery, (recovery, offset, height))
                self.assertFalse(trunc)

    def test_recovery_stops_on_floor_contact_or_escape(self):
        e = BalanceEnv(recovery_mode=True, max_steps=None)
        self.addCleanup(e.close)
        for pos, expected in (([.39, 0., .021], 'ground_contact'),
                              ([1.2, 0., .5], 'beyond_recovery_distance')):
            e.reset(seed=0)
            e.d.qpos[e.ball_qadr:e.ball_qadr+3] = pos
            e.d.qvel[e.ball_vadr:e.ball_vadr+6] = 0
            mujoco.mj_forward(e.m, e.d)
            _, _, term, _, info = e.step(np.zeros(6))
            self.assertTrue(term)
            self.assertEqual(info['termination_reason'], expected)

    def test_continuous_mode_does_not_reset_at_ten_seconds(self):
        e = BalanceEnv(recovery_mode=True, max_steps=None,
                       ball_start_radius=0, ball_start_speed=0)
        self.addCleanup(e.close)
        e.reset(seed=0)
        e.t = 299
        _, _, term, trunc, info = e.step(np.zeros(6))
        self.assertFalse(term or trunc)
        self.assertFalse(info['is_success'])

    def test_urdf_limits_and_effort(self):
        source = SOURCE_URDF
        root = ET.parse(source).getroot()
        e = BalanceEnv()
        self.addCleanup(e.close)
        self.assertEqual(e.m.nu, 6)
        for i, name in enumerate(ARM_JOINTS):
            limit = root.find(f"joint[@name='{name}']/limit")
            np.testing.assert_allclose([e.q_lo[i], e.q_hi[i]], [float(limit.get('lower')), float(limit.get('upper'))], atol=5e-6)
            effort = float(limit.get('effort'))
            np.testing.assert_allclose(e.m.actuator_forcerange[i], [-effort, effort])

    def test_level_ball_contact_and_hold(self):
        e = BalanceEnv(ball_start_radius=0, ball_start_speed=0, act_hist=3)
        self.addCleanup(e.close)
        e.reset(seed=0)
        e.d.qpos[e.q_adr] = HOME_Q
        e._q_target = HOME_Q.copy()
        e.d.ctrl[e.act_ids] = HOME_Q
        mujoco.mj_forward(e.m, e.d)
        c, n, u, w = e.blade_frame()
        np.testing.assert_allclose(n, [0,0,1], atol=1e-6)
        np.testing.assert_allclose(np.array([u,w,n]) @ np.array([u,w,n]).T, np.eye(3), atol=1e-6)
        e.d.qpos[e.ball_qadr:e.ball_qadr+3] = c + n * (e.ball_radius + .011)
        mujoco.mj_forward(e.m,e.d)
        contacts = False
        for _ in range(300):
            obs, _, term, trunc, info = e.step(np.zeros(6))
            contacts |= any({con.geom1, con.geom2} == {e.ball_geom,e.blade_geom} for con in e.d.contact)
            self.assertFalse(term)
        self.assertTrue(contacts)
        self.assertTrue(trunc and info['is_success'])
        self.assertEqual(obs.shape, (34,))
        np.testing.assert_allclose(e.d.qpos[e.q_adr], HOME_Q, atol=0.002)
        self.assertFalse(np.any(e.d.warning.number))

    def test_randomized_camera_and_action(self):
        e = BalanceEnv(obs_mode='image', domain_rand=True, servo_rand=True, ball_noise=1, act_hist=3)
        self.addCleanup(e.close)
        obs,_=e.reset(seed=3)
        self.assertEqual(obs['image'].shape,(96,128,3))
        self.assertGreater(obs['image'].std(), 5)
        with self.assertRaises(ValueError):
            e.step(np.zeros(5))
        for _ in range(10):
            obs,_,term,trunc,_=e.step(np.full(6,0.01))
            self.assertTrue(np.isfinite(e.controller_obs()).all())
            if term or trunc:
                e.reset()
        self.assertFalse(np.any(e.d.warning.number))

    def test_d405_mount_and_ball_visibility(self):
        from perception.detector import detect
        e = BalanceEnv(obs_mode='image')
        self.addCleanup(e.close)
        source = SOURCE_URDF
        origin = ET.parse(source).getroot().find("joint[@name='camera_joint']/origin")
        camera_body = e.m.body('camera_link')
        self.assertEqual(e.m.cam_bodyid[e.cam_id], camera_body.id)
        self.assertAlmostEqual(float(camera_body.mass[0]), .08)
        np.testing.assert_allclose(camera_body.pos, np.fromstring(origin.get('xyz'), sep=' '), atol=1e-6)
        for seed in range(3):
            obs, _ = e.reset(seed=seed)
            self.assertEqual(detect(obs['image'])[3], 1)

    def test_d405_detector_rejects_wooden_handle(self):
        from perception.detector import detect
        e = BalanceEnv(obs_mode='image', domain_rand=True, cam_rand=0)
        self.addCleanup(e.close)
        for seed in [10004, 10006, 10008]:
            obs, _ = e.reset(seed=seed)
            detection = detect(obs['image'])
            self.assertEqual(detection[3], 1)
            local = e.d.cam_xmat[e.cam_id].reshape(3,3).T @ (e.d.xpos[e.ball_body] - e.d.cam_xpos[e.cam_id])
            tangent = np.tan(np.deg2rad(e.m.cam_fovy[e.cam_id]) / 2)
            expected = np.array([local[0]/(-local[2]*tangent*(128/96)), -local[1]/(-local[2]*tangent)])
            np.testing.assert_allclose(detection[:2], expected, atol=.07)

    def test_vision_control_does_not_read_true_ball_state(self):
        from unittest.mock import Mock, patch
        from perception.vision_control import VisionController
        from sim.env import FdVel
        e = BalanceEnv()
        self.addCleanup(e.close)
        e.reset(seed=1)
        e.controller_obs = Mock(side_effect=AssertionError('true ball state accessed'))
        vision = VisionController.__new__(VisionController)
        vision.controller = object()
        vision.estimator = Mock()
        vision.estimator.update.return_value = np.array([.1,.2,0,0])
        vision.fd = FdVel()
        with patch('perception.vision_control.controller_action', return_value=np.zeros(6)) as action:
            vision.action(e, np.zeros((96,128,3), np.uint8))
            np.testing.assert_allclose(action.call_args.args[1][:4], [.1,.2,0,0])
        e.controller_obs.assert_not_called()

    def test_closed_deep_grip_and_estimator_frame(self):
        import torch
        from perception.estimator import EstRunner
        from sim.robot import PADDLE_INSERTION_M, GRIPPER_GAP_M
        e = BalanceEnv()
        self.addCleanup(e.close)
        self.assertAlmostEqual(e.m.body('gripper_L_link').pos[1] - e.m.body('gripper_R_link').pos[1], GRIPPER_GAP_M)
        self.assertAlmostEqual(e.m.body('gripper').pos[0], -.068 - PADDLE_INSERTION_M)
        class OldEstimator(torch.nn.Module):
            paddle_insertion_m = 0.0
            def forward(self, x):
                return torch.zeros((len(x),4))
        runner = EstRunner(OldEstimator())
        expected = PADDLE_INSERTION_M / .08
        self.assertAlmostEqual(float(runner.update([0,0,.7,1])[1]), expected)
        self.assertAlmostEqual(float(runner.update([0,0,0,0])[1]), expected)
        # The model trained on this insertion depth needs no offset.
        model = OldEstimator()
        model.paddle_insertion_m = PADDLE_INSERTION_M
        self.assertAlmostEqual(float(EstRunner(model).update([0,0,.7,1])[1]), 0.)

    def test_reject_so101_weights(self):
        import tempfile
        import torch
        from perception.estimator import load_est
        with tempfile.TemporaryDirectory() as folder:
            policy = Path(folder)/'old_policy.pt'
            estimator = Path(folder)/'old_estimator.pt'
            torch.save({'robot_id':'so101', 'model':{'actor.0.weight':torch.zeros(1,22),
                                                   'log_std':torch.zeros(5)}}, policy)
            torch.save({'robot_id':'so101'}, estimator)
            with self.assertRaisesRegex(ValueError, 'not a Panthera'):
                load_controller(policy)
            with self.assertRaisesRegex(ValueError, 'not calibrated for Panthera'):
                load_est(estimator)


if __name__ == '__main__':
    unittest.main()
