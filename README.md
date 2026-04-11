# OmniGrasp

[![OmniGrasp CI](https://github.com/Jayakshata/omnigrasp/actions/workflows/ci.yml/badge.svg)](https://github.com/Jayakshata/omnigrasp/actions/workflows/ci.yml)

**Sim-to-real VLM + RL robotic manipulation pipeline** with a production-grade, multi-model perception stack.

A closed-loop system where a human types a natural language command, and a robot arm in simulation sees the target object, plans a grasp, and executes it — all without being explicitly programmed for that specific object.

## Status

🚧 **Under active development** — Week 1 of 4

## Tech Stack

- **Perception:** Grounding DINO + OWL-ViT (multi-model fusion), SAM 2 (segmentation)
- **3D Vision:** Pinhole camera model, depth deprojection, frame transforms
- **Tracking:** Kalman filter (6-state temporal smoothing)
- **Control:** PPO (Proximal Policy Optimization) via Stable Baselines3
- **Simulation:** NVIDIA Isaac Sim + Isaac Lab
- **Middleware:** ROS2 Humble Hawksbill
- **Infrastructure:** Docker, Docker Compose, GitHub Actions CI/CD
- **Frontend:** PyScript + rosbridge WebSocket