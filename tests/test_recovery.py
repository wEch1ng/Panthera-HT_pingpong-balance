import unittest
from unittest.mock import patch
import numpy as np
import mujoco
from sim.recovery_env import RecoveryEnv
from sim.env import BalanceEnv
from perception.rgbd_ball import RGBDBallTracker


class RecoveryTests(unittest.TestCase):
    def test_rgbd_measures_airborne_height_without_truth_observation(self):
        e = RecoveryEnv(act_hist=3, domain_rand=False, recovery_level=0.)
        self.addCleanup(e.close)
        e.reset(seed=0)
        c,n,u,w = e.blade_frame()
        for h in (.03,.07):
            e.d.qpos[e.ball_qadr:e.ball_qadr+3] = c+n*h
            mujoco.mj_forward(e.m,e.d)
            e._sensor_time = None
            e.tracker.reset()
            with (patch.object(e,'ball_in_blade',side_effect=AssertionError('truth observation')),
                  patch.object(BalanceEnv,'controller_obs',side_effect=AssertionError('truth observation'))):
                obs=e.controller_obs()
            self.assertEqual(obs.shape,(42,))
            self.assertEqual(obs[36],1.)
            self.assertAlmostEqual(obs[34]*.1,h,delta=.008)
            np.testing.assert_allclose(obs[:2]*.08,0.,atol=.008)

    def test_mechanical_energy_penalty_and_urdf_limits(self):
        envs=[RecoveryEnv(energy_weight=k,domain_rand=False,recovery_level=0.) for k in (0.,.04)]
        for e in envs:
            self.addCleanup(e.close)
            e.reset(seed=0)
        outcomes=[e.step(np.full(6,.3)) for e in envs]
        power=outcomes[0][4]['power_w']
        self.assertGreater(power,0.)
        self.assertAlmostEqual(outcomes[0][1]-outcomes[1][1],.04*power,places=5)
        for e, result in zip(envs,outcomes):
            self.assertAlmostEqual(result[4]['energy_j'],power/30,places=6)
            self.assertTrue(np.all(e._q_target>=e.q_lo) and np.all(e._q_target<=e.q_hi))
            self.assertTrue(np.all(np.abs(e.d.actuator_force[e.act_ids]) <= e.m.actuator_forcerange[e.act_ids,1]+1e-6))

    def test_velocity_violation_terminates(self):
        e=RecoveryEnv(domain_rand=False,recovery_level=0.)
        self.addCleanup(e.close)
        e.reset(seed=0)
        e.d.qvel[e.v_adr[0]]=e.q_speed_limits[0]*1.5
        _,_,term,_,info=e.step(np.zeros(6))
        self.assertTrue(term)
        self.assertEqual(info['termination_reason'],'speed_limit')

    def test_occlusion_flag_and_bounded_extrapolation(self):
        tracker=RGBDBallTracker()
        tracker.update(np.array([0.,0.,.5]))
        p,v,vis,age=tracker.update(None)
        self.assertEqual(vis,0)
        self.assertLess(p[2],.5)
        for _ in range(30): tracker.update(None)
        before=tracker.position.copy()
        tracker.update(None)
        np.testing.assert_array_equal(before,tracker.position)


if __name__=='__main__':
    unittest.main()
