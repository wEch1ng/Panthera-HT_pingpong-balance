"""Independent RGB-D recovery evaluation, same seeds/scenarios for each policy.

python -m sim.eval_recovery --controller CHECKPOINT --episodes 5 --json RESULT
Episodes are per scenario, each 20 seconds. No privileged policy observations.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from sim.policies import load_controller, controller_action
from sim.recovery_env import RecoveryEnv


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--controller',required=True)
    p.add_argument('--episodes',type=int,default=5)
    p.add_argument('--level',type=float,default=1.)
    p.add_argument('--seed',type=int,default=20000)
    p.add_argument('--latency',type=int,default=2)
    p.add_argument('--json',required=True)
    a=p.parse_args()
    torch.set_num_threads(1)
    ac=load_controller(a.controller)
    if not ac.train_args.get('recovery_task'):
        p.error('Use a recovery checkpoint or its initial.pt for baseline comparison')
    ta=ac.train_args
    env=RecoveryEnv(act_hist=ac.act_hist,domain_rand=True,max_latency=a.latency,
                    recovery_level=a.level,**{k:ta[k] for k in ('target_height','target_u','target_w','energy_weight')})
    rows=[]
    for scenario in ('balance','rolling','drop','low_return'):
        for i in range(a.episodes):
            obs,_=env.reset(seed=a.seed+i,options={'scenario':scenario})
            env._action_delay=a.latency
            env._delay_buf=[np.zeros(6) for _ in range(a.latency)]
            initial_z=float(env.blade_frame()[0][2])
            visible_frames=0
            airborne_seen=False
            recaught=False
            max_stable=0
            for t in range(env.max_steps):
                visible_frames+=int(obs[-6]>0)
                obs,r,term,trunc,info=env.step(controller_action(ac,obs))
                if not info['caught'] and info['ball_h']>.05:
                    airborne_seen=True
                recaught |= airborne_seen and info['caught']
                max_stable=max(max_stable,info['stable_steps'])
                if term or trunc:break
            row=dict(scenario=scenario,seed=a.seed+i,success=info['is_success'],
                     duration_s=(t+1)/30,energy_j=info['energy_j'],
                     height_error_m=info['height_error_m'],ball_center_error_m=float(np.hypot(info['ball_u']-ta['target_u'],info['ball_w']-ta['target_w'])),
                     lowered_from_start_m=initial_z-info['min_blade_height_m'],
                     airborne_seen=bool(airborne_seen),recaught=bool(recaught),
                     reached_goal_for_one_second=bool(max_stable>=30),
                     visible_fraction=visible_frames/(t+1),reason=info['termination_reason'])
            rows.append(row)
            print(json.dumps(row),flush=True)
    env.close()
    result=dict(controller=a.controller,level=a.level,latency=a.latency,
                criteria='20 s without failure, last 1 s: ball within 2 cm of target, blade height within 2 cm, contact and ball speed <0.08 m/s',
                successes=sum(r['success'] for r in rows),episodes=rows)
    output = Path(a.json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result,indent=2))
    print(f"Success: {result['successes']}/{len(rows)}",flush=True)


if __name__=='__main__':
    main()
