# Checkpoints

| File | Use |
| --- | --- |
| `d405_rgbd_recovery.pt` | Final 1500-iteration RGB-D recovery policy, 42 inputs; no MLP estimator argument |
| `d405_rgbd_recovery_initial.pt` | Expanded initial policy before recovery training, for baseline evaluation |
| `d405_rgb_balance.pt` | 6000-iteration D405 noise-adapted balance policy, 34 inputs; recovery initializer |
| `d405_rgb_estimator.pt` | RGB-only MLP estimator for the balance policy |

Optimizer states are excluded. Use these for inference or `--init`; use a complete `outputs/controller/<run>/last.pt` for `--resume`. `manifest.json` records SHA-256 and original experiment-relative source paths. Model and camera assumptions are documented in the root README.
