# Getting Started with AdaptiveRL

Welcome to AdaptiveRL! This guide will help you get up and running quickly.

## Installation

Ensure you have completed the installation steps described in the main `README.md`.

## System Diagnostics

Immediately after your initial installation, we highly recommend running the `adaptive-rl doctor` command. This will perform a multi-point diagnostic check of your local environment to ensure that everything is configured correctly before you begin training your drone.

```bash
adaptive-rl doctor
```

The system doctor will scan your:
- Python version
- PyTorch GPU/MPS acceleration
- Core RL dependencies
- Gymnasium environment registrations
- Hardware thread count
- Model checkpoint capabilities
- YAML configuration structural integrity

If any step fails, the doctor will provide you with the exact command to fix the issue.

## Next Steps

Once the doctor gives you a clean bill of health, proceed to:

1. **Verify environments**: `adaptive-rl env inspect drone`
2. **Train your first agent**: `adaptive-rl train --config configs/drone_ppo_demo.yaml`
3. **Run the 3D visualizer**: `adaptive-rl gui`

Happy flying!
