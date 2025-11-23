# Reinforcement Learning for Autonomous Vehicle Navigation in Factory Environments

[![ROS 2 Humble](https://img.shields.io/badge/ROS%202-Humble-blue)](https://docs.ros.org/en/humble/)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-green)](https://www.python.org/)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-PPO-orange)](https://stable-baselines3.readthedocs.io/)

## Overview

This repository contains the complete implementation and results of my thesis on **Deep Reinforcement Learning for Autonomous Navigation** using a PiCar-X robot platform. The project demonstrates how PPO (Proximal Policy Optimization) can be trained to navigate complex factory environments using multi-modal sensor inputs.

**Key Features:**
- Complete ROS 2 Humble workspace with custom RL environment
- Gazebo Harmonic simulation with factory world models
- Multi-modal sensor fusion (camera, ultrasonic, IR line sensors)
- Trained PPO model (1M timesteps) with evaluation results
- Comprehensive documentation and thesis PDF

**Important Note:** All training was conducted using **MOCK mode** (kinematic simulation) for rapid prototyping and algorithm validation. The Gazebo integration infrastructure is included but ROS mode integration is planned future work.

---

## Thesis Document

The complete thesis document is available at **[thesis/Thesis_RL_Final.pdf](thesis/Thesis_RL_Final.pdf)** .

**Abstract:** This research investigates the application of deep reinforcement learning for autonomous navigation in industrial factory environments. Using the Proximal Policy Optimization (PPO) algorithm, I trained an agent to navigate using camera images, ultrasonic range data, and IR line sensors in simulation. The approach demonstrates the potential for learned navigation policies that can adapt to dynamic environments without explicit path planning.

---

## Repository Structure

```
rl-factory-thesis/
├── README.md                    # This file
├── LICENSE                      
├── .gitignore                   
│
├── ros2_ws/                     # ROS 2 Workspace
│   └── src/
│       ├── factory_nav_rl/      # Main RL package
│       │   ├── env_factory_nav.py      # Gymnasium environment
│       │   ├── train_ppo.py             # Training script
│       │   ├── evaluate.py              # Evaluation script
│       │   └── plot_results.py          # Visualization tools
│       ├── picarx_description/  # Robot URDF/SDF models
│       └── picarx_gz/           # Gazebo worlds and launch files
│
├── thesis/                      # Thesis Document
│   ├── Thesis_RL_Final.pdf         # Compiled thesis (2.6MB)
│   └── figures/                 # Thesis figures
│
├── models/                      # Trained Models
│   └── ppo_factory_nav_final.zip   # 1M timestep trained PPO model (8.7MB)
│
└── results/                     # Evaluation Results
    ├── evaluation_results/      # Trained model performance data
    ├── baseline_results/        # Random policy baseline
    └── figures/                 # Training curves and comparisons
```

---

## Quick Start (3 Steps)

### Prerequisites

- **Operating System:** Ubuntu 22.04 LTS
- **ROS 2:** Humble Hawksbill ([installation guide](https://docs.ros.org/en/humble/Installation.html))
- **Python:** 3.12 or higher
- **Gazebo:** Harmonic v8 (optional for visualization)

### 1. Clone the Repository

```bash
git clone https://github.com/luisglm1/rl-factory-thesis.git
cd rl-factory-thesis
```

### 2. Set Up the Environment

```bash
# Create and activate Python virtual environment
python3 -m venv rl-venv
source rl-venv/bin/activate

# Install Python dependencies
pip install stable-baselines3 gymnasium numpy matplotlib tensorboard

# Build ROS 2 workspace
cd ros2_ws
colcon build
source install/setup.bash
cd ..
```

### 3. Run Evaluation with Pre-trained Model

```bash
# Activate environment
source rl-venv/bin/activate
source ros2_ws/install/setup.bash

# Run quick evaluation (10 episodes)
python3 ros2_ws/src/factory_nav_rl/factory_nav_rl/evaluate.py \
  --model models/ppo_factory_nav_final.zip \
  --episodes 10

# Results will be printed to console
```


---

## Detailed Usage Guide

### Training from Scratch

The environment supports two simulation modes:

#### **MOCK Mode (Recommended, Fast)**

Pure Python kinematic simulation without Gazebo. Best for rapid experimentation and algorithm development.

```bash
# Activate environments
source rl-venv/bin/activate
source ros2_ws/install/setup.bash

# Train with default settings (10k timesteps for testing)
python3 ros2_ws/src/factory_nav_rl/factory_nav_rl/train_ppo.py

# Full training run (1M timesteps, ~2-4 hours on modern CPU)
python3 ros2_ws/src/factory_nav_rl/factory_nav_rl/train_ppo.py \
  --timesteps 1000000 \
  --n_envs 8 \
  --log_dir ~/my_training \
  --save_freq 50000

# Monitor training with TensorBoard
tensorboard --logdir ~/my_training/tensorboard
```

**Training Hyperparameters:**
- Algorithm: PPO (Proximal Policy Optimization)
- Policy: MultiInputPolicy (handles dict observations)
- Parallel Environments: 4-8 (CPU-based)
- Learning Rate: 3e-4 (default)
- Batch Size: 64
- Clip Range: 0.2

#### **ROS Mode with Gazebo (Future Work)**

Full physics simulation with Gazebo Harmonic. **Note:** This mode is not fully implemented yet (see limitations below).

```bash
# Terminal 1: Launch Gazebo simulation
source ros2_ws/install/setup.bash
python3 ros2_ws/src/picarx_gz/picarx_gz/launch_world.py

# Terminal 2: Train with ROS mode (when implemented)
source rl-venv/bin/activate
source ros2_ws/install/setup.bash
python3 ros2_ws/src/factory_nav_rl/factory_nav_rl/train_ppo.py \
  --mode ros \
  --timesteps 1000000
```

---

### Evaluating Trained Models

```bash
# Comprehensive evaluation (100 episodes)
python3 ros2_ws/src/factory_nav_rl/factory_nav_rl/evaluate.py \
  --model models/ppo_factory_nav_final.zip \
  --episodes 100 \
  --output ~/eval_results

# Compare with random baseline
python3 ros2_ws/src/factory_nav_rl/factory_nav_rl/evaluate_random.py \
  --episodes 100 \
  --output ~/baseline_results
```

**Evaluation Metrics:**
- **Success Rate:** % of episodes reaching the goal
- **Collision Rate:** % of episodes ending in collision
- **Timeout Rate:** % of episodes exceeding max steps (500)
- **Path Efficiency:** Ratio of optimal distance to actual distance traveled
- **Mean Episode Reward:** Average cumulative reward

---

### Reproducing Thesis Results

```bash
# Generate all thesis figures from training logs
python3 ros2_ws/src/factory_nav_rl/factory_nav_rl/plot_results.py \
  --log_dir path/to/training/tensorboard \
  --eval_dir path/to/eval_results \
  --output thesis_figures/

# Figures will be saved as PNG/PDF in the output directory
```

---

## System Architecture

### Observation Space (Multi-Modal)

The RL agent receives observations as a **Dict**:

```python
{
  "camera": Box(0, 255, shape=(80, 60, 3), dtype=uint8),     # RGB camera image
  "ultrasonic": Box(0.0, 1.0, shape=(8,), dtype=float32),    # Normalized range data
  "ir_sensors": Box(0.0, 1.0, shape=(3,), dtype=float32)     # IR line tracking sensors
}
```

### Action Space

Continuous control with 2 dimensions:

```python
Box(low=[0.0, -1.0], high=[0.5, 1.0], dtype=float32)
# [throttle, steering]
# throttle: 0.0 to 0.5 m/s
# steering: -1.0 (full left) to 1.0 (full right)
```

### Reward Function

```python
reward = -distance_to_goal      # Encourages approaching goal
         - 50 * collision        # Heavy penalty for crashes
         - 0.1 * timestep        # Small time penalty for efficiency
         + 100 * goal_reached    # Large bonus for success
```

### Data Flow (MOCK Mode)

```
Python Kinematic Simulation (env_factory_nav.py)
    ↓ [synthetic observations: camera gradients, simulated lidar, IR sensors]
PPO Agent (MultiInputPolicy)
    ↓ [actions: throttle, steering]
Kinematic Model (differential drive)
    ↓ [pose update: x, y, θ]
Collision Detection & Goal Check
    ↓ [reward, done, info]
Agent Update (PPO optimization)
```

---

## Pre-trained Model Performance

The included model (`models/ppo_factory_nav_final.zip`) was trained for **1 million timesteps** in MOCK mode.

**Results (100 episodes):**
- **Mean Episode Reward:** +185 ± 67
- **Success Rate:** 22% (conservative policy, needs longer training)
- **Collision Rate:** 0% (successfully learned collision avoidance)
- **Timeout Rate:** 78% (reaches max steps without goal)
- **Mean Episode Length:** 500 steps (max_steps limit)

**Analysis:**
The agent exhibits **overly conservative behavior** - it successfully avoids all obstacles but does not reach the goal. This suggests:
1. Reward function tuning needed (increase goal bonus, reduce time penalty)
2. Longer training duration (2-5M timesteps recommended)
3. Curriculum learning (start with easier goals, gradually increase difficulty)

See **[thesis/Thesis_RL_Final.pdf](thesis/Thesis_RL_Final.pdf)** for detailed analysis and comparison with baselines.

---

## Configuration and Customization

### Environment Parameters

Edit `ros2_ws/src/factory_nav_rl/factory_nav_rl/env_factory_nav.py`:

```python
# Navigation
goal_radius: 0.3          # Success threshold (meters)
max_steps: 500            # Episode timeout
world_bounds: (-10, 10)   # Environment size (meters)

# Sensors
camera_resolution: (80, 60)
ultrasonic_beams: 8
ultrasonic_max_range: 2.5  # meters

# Reward Shaping
collision_penalty: 50.0
goal_bonus: 100.0
time_penalty: 0.1
```

### Training Hyperparameters

Edit `ros2_ws/src/factory_nav_rl/factory_nav_rl/train_ppo.py`:

```python
# PPO Settings
learning_rate: 3e-4
n_steps: 2048
batch_size: 64
n_epochs: 10
gamma: 0.99
clip_range: 0.2

# Training
total_timesteps: 1000000
n_envs: 8  # Parallel environments
save_freq: 50000  # Checkpoint frequency
```

---

## Known Limitations and Future Work

### Current Limitations

1. **Gazebo ROS Integration:** Not fully implemented
   - MOCK mode is fully functional and was used for all thesis results
   - ROS mode has placeholder code but requires additional work

2. **Model Performance:** Conservative behavior
   - 22% success rate on goal reaching (but 0% collisions)
   - Needs longer training or reward tuning

3. **Sim-to-Real Transfer:** Not yet tested
   - Deployment to physical PiCar-X is future work
   - Domain randomization not implemented in Gazebo

### Planned Future Work

- Complete ROS-Gazebo integration for real-time sensor data
- Implement domain randomization (lighting, textures, obstacles)
- Extend training to 5M+ timesteps
- Tune reward function for better goal-reaching behavior
- Deploy to physical PiCar-X robot
- Compare with traditional navigation methods (DWA, TEB)
- Add support for dynamic obstacles

**Contributions are welcome!** 

---

## Dependencies

### System Requirements

- **Ubuntu:** 22.04 LTS
- **ROS 2:** Humble Hawksbill
- **Gazebo:** Harmonic v8 
- **Python:** 3.12+

### Python Packages

```
stable-baselines3>=2.0.0
gymnasium>=0.29.0
numpy>=1.24.0
matplotlib>=3.7.0
tensorboard>=2.13.0
opencv-python>=4.8.0
```

Install with:
```bash
pip install stable-baselines3 gymnasium numpy matplotlib tensorboard opencv-python
```

### ROS 2 Dependencies

```bash
sudo apt install ros-humble-ros-gz-sim ros-humble-ros-gz-bridge
```

---

## Troubleshooting

### Build Errors

**Problem:** `colcon build` fails with missing dependencies

```bash
# Install missing ROS 2 packages
sudo apt update
rosdep install --from-paths src --ignore-src -r -y

# Clean and rebuild
cd ros2_ws
rm -rf build/ install/ log/
colcon build
```

### Python Import Errors

**Problem:** `ModuleNotFoundError: No module named 'stable_baselines3'`

```bash
# Ensure virtual environment is activated
source rl-venv/bin/activate

# Reinstall dependencies
pip install --upgrade stable-baselines3 gymnasium numpy
```

### Training Not Improving

**Problem:** Agent stuck with low rewards, no progress

**Solutions:**
1. Increase training timesteps (`--timesteps 5000000`)
2. Tune reward function (increase `goal_bonus`, decrease `time_penalty`)
3. Adjust learning rate (`--learning_rate 1e-4`)
4. Use curriculum learning (start with closer goals)

### Gazebo Simulation Issues

**Problem:** Sensors not publishing data in Gazebo

**Solution:** See the Gazebo Documentation or Pray 🙏

---


## License

This project is licensed under the **MIT License** - see the LICENSE file for details.

**You are free to:**
- Use this code for academic research
- Modify and build upon this work
- Use in commercial applications (with attribution)

---

## Acknowledgments

- **Stable-Baselines3** team for the excellent RL library
- **ROS 2** and **Gazebo** communities for simulation tools
- **PiCar-X** platform for the robot hardware reference
- Thesis advisor and academic reviewers

---

## Contact and Support

- **Issues:** Please open a GitHub issue for bugs or questions
- **Discussions:** Use GitHub Discussions for general questions

---

**Last Updated:** November 2025
**Version:** 1.0.0
**Status:** Thesis Completed, Open for Contributions
