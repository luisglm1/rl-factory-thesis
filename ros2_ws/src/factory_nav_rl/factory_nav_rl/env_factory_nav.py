"""
Factory Navigation RL Environment
Supports both MOCK mode (simulated physics) and ROS mode (real Gazebo integration)
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math

class FactoryNavEnv(gym.Env):
    """
    Factory navigation environment for PPO training.

    Observation space:
        - image: 80x60x3 RGB camera (or 60x80x3 based on init)
        - scan: Single float representing minimum distance from 61-ray lidar
        - ir: 3-binary sensors for line tracking (not used in Phase 1)

    Action space:
        - throttle: [0.0, 0.5] forward velocity
        - steering: [-1.0, 1.0] angular velocity

    Args:
        width: Image width (default 80)
        height: Image height (default 60)
        mode: 'mock' for simulated physics, 'ros' for Gazebo integration
        max_steps: Maximum steps per episode
    """
    metadata = {"render_modes": []}

    def __init__(self, width=80, height=60, mode='mock', max_steps=1000, gamma=0.995, curriculum_stage='full'):
        super().__init__()
        self.img_h, self.img_w = height, width
        self.mode = mode
        self.max_steps = max_steps
        self.gamma = gamma  # Discount factor for potential-based shaping
        self.curriculum_stage = curriculum_stage  # 'simple', 'medium', or 'full'

        # Observation space
        self.observation_space = spaces.Dict({
            "image": spaces.Box(0, 255, (self.img_h, self.img_w, 3), dtype=np.uint8),
            "scan":  spaces.Box(0.0, 4.0, (1,), dtype=np.float32),
            "ir":    spaces.MultiBinary(3),
        })

        # Action space: [throttle, steering]
        # PHASE 1A FIX: Increased throttle from [0.0, 0.5] to [0.0, 1.0] for full speed
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),  # Full throttle range
            dtype=np.float32
        )

        # Episode tracking
        self.step_count = 0
        self.episode_count = 0

        # Reward computation tracking
        self._prev_dist_to_goal = None
        self._prev_action = None

        # MOCK MODE: Simulated state
        if self.mode == 'mock':
            self._init_mock_state()
        else:
            # ROS MODE: Initialize ROS node and subscribers
            self._init_ros()

    def _init_mock_state(self):
        """Initialize mock simulation state"""
        # Robot pose in 2D: [x, y, theta]
        self.robot_pose = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Curriculum-based configuration
        if self.curriculum_stage == 'simple':
            # Stage 1: Short distance, no obstacles (learn goal-seeking)
            self.goal_distance_range = (2.0, 3.0)
            self.goal_radius = 0.6  # Slightly larger for easier success
            self.obstacles = np.array([])  # No obstacles
            print(f"[FactoryNavEnv] Initialized in MOCK mode - CURRICULUM STAGE 1: SIMPLE")
        elif self.curriculum_stage == 'medium':
            # Stage 2: Medium distance, 1 obstacle (learn avoidance)
            self.goal_distance_range = (4.0, 6.0)
            self.goal_radius = 0.5
            self.obstacles = np.array([
                [3.0, 0.0, 0.5],   # Single obstacle in path
            ])
            print(f"[FactoryNavEnv] Initialized in MOCK mode - CURRICULUM STAGE 2: MEDIUM")
        else:  # 'full'
            # Stage 3: Full distance, 3 obstacles (original task)
            self.goal_distance_range = (3.5, 4.5)
            self.goal_radius = 0.5
            self.obstacles = np.array([
                [2.0, 1.0, 0.5],   # Obstacle 1
                [3.0, -1.0, 0.6],  # Obstacle 2
                [1.5, 0.0, 0.4],   # Obstacle 3
            ])
            print(f"[FactoryNavEnv] Initialized in MOCK mode - CURRICULUM STAGE 3: FULL")

        # Goal position (will be set in reset())
        self.goal_pos = np.array([3.0, 0.0], dtype=np.float32)

        # Velocity (for realism)
        self.robot_velocity = np.array([0.0, 0.0], dtype=np.float32)  # [linear, angular]

    def _init_ros(self):
        """Initialize ROS 2 node and subscribers (for future implementation)"""
        print(f"[FactoryNavEnv] ROS mode not yet implemented, falling back to MOCK")
        self.mode = 'mock'
        self._init_mock_state()

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        self.step_count = 0
        self.episode_count += 1

        # Reset reward tracking
        self._prev_dist_to_goal = None
        self._prev_action = None

        if self.mode == 'mock':
            # Random start position (curriculum-adapted)
            if self.curriculum_stage == 'simple':
                # Start at origin for simplicity
                self.robot_pose = np.array([
                    0.0,
                    0.0,
                    np.random.uniform(-np.pi/4, np.pi/4)  # Roughly facing goal
                ], dtype=np.float32)
            else:
                # Random start position (avoid obstacles)
                self.robot_pose = np.array([
                    np.random.uniform(-1.0, 0.0),
                    np.random.uniform(-0.5, 0.5),
                    np.random.uniform(-np.pi, np.pi)
                ], dtype=np.float32)

            # Random goal position (curriculum-based distance)
            goal_dist = np.random.uniform(*self.goal_distance_range)
            goal_angle = np.random.uniform(-np.pi/3, np.pi/3)  # Forward cone
            self.goal_pos = np.array([
                self.robot_pose[0] + goal_dist * np.cos(goal_angle),
                self.robot_pose[1] + goal_dist * np.sin(goal_angle)
            ], dtype=np.float32)

            self.robot_velocity = np.array([0.0, 0.0], dtype=np.float32)

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action):
        """Execute one step in the environment"""
        self.step_count += 1

        # Parse action
        throttle = float(action[0])  # [0.0, 1.0] - UPDATED for full speed
        steering = float(action[1])  # [-1.0, 1.0]

        if self.mode == 'mock':
            # Update mock physics (simple kinematic model)
            dt = 0.1  # 10 Hz simulation

            # Update velocities (with smoothing)
            # Max velocity now 1.0 m/s (doubled from 0.5)
            target_linear = throttle * 1.0  # Max 1.0 m/s
            target_angular = steering * 1.5  # Max 1.5 rad/s

            self.robot_velocity[0] = 0.7 * self.robot_velocity[0] + 0.3 * target_linear
            self.robot_velocity[1] = 0.7 * self.robot_velocity[1] + 0.3 * target_angular

            # Update pose
            self.robot_pose[0] += self.robot_velocity[0] * np.cos(self.robot_pose[2]) * dt
            self.robot_pose[1] += self.robot_velocity[0] * np.sin(self.robot_pose[2]) * dt
            self.robot_pose[2] += self.robot_velocity[1] * dt

            # Normalize angle to [-pi, pi]
            self.robot_pose[2] = np.arctan2(np.sin(self.robot_pose[2]), np.cos(self.robot_pose[2]))

        # Get observation
        obs = self._get_observation()

        # Compute reward
        reward = self._compute_reward(action)

        # Check termination
        terminated = self._check_termination()
        truncated = self.step_count >= self.max_steps

        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _get_observation(self):
        """Get current observation"""
        if self.mode == 'mock':
            # Generate synthetic camera image (simple gradient based on proximity)
            image = self._generate_mock_image()

            # Compute scan (min distance to obstacles)
            scan_distance = self._compute_mock_scan()

            # IR sensors (placeholder)
            ir_sensors = np.array([0, 0, 0], dtype=np.int8)

            return {
                "image": image,
                "scan": np.array([scan_distance], dtype=np.float32),
                "ir": ir_sensors
            }
        else:
            # ROS mode: return actual sensor data (TODO)
            pass

    def _generate_mock_image(self):
        """Generate a synthetic camera image"""
        # Create a simple gradient image based on goal direction
        image = np.ones((self.img_h, self.img_w, 3), dtype=np.uint8) * 128

        # Add goal-direction indicator (brighter on side of goal)
        goal_angle = np.arctan2(
            self.goal_pos[1] - self.robot_pose[1],
            self.goal_pos[0] - self.robot_pose[0]
        )
        angle_diff = goal_angle - self.robot_pose[2]
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

        # Create horizontal gradient pointing toward goal
        x = np.linspace(-1, 1, self.img_w)
        gradient = np.clip((x * np.sign(angle_diff) + 1) * 127, 0, 255).astype(np.uint8)
        image[:, :, 1] = gradient[np.newaxis, :]  # Green channel shows direction

        # Add noise for realism
        noise = np.random.randint(0, 30, (self.img_h, self.img_w, 3), dtype=np.uint8)
        image = np.clip(image.astype(np.int16) + noise - 15, 0, 255).astype(np.uint8)

        return image

    def _compute_mock_scan(self):
        """Compute minimum distance to obstacles (simulated lidar)"""
        min_dist = 4.0  # Max range

        # Check distance to each obstacle
        for obs in self.obstacles:
            obs_x, obs_y, obs_r = obs
            dist = np.sqrt((self.robot_pose[0] - obs_x)**2 + (self.robot_pose[1] - obs_y)**2)
            dist = max(0.03, dist - obs_r)  # Subtract obstacle radius
            min_dist = min(min_dist, dist)

        # Check boundaries (simple -5 to 5 in x and y)
        dist_to_bounds = min(
            5.0 - abs(self.robot_pose[0]),
            5.0 - abs(self.robot_pose[1])
        )
        min_dist = min(min_dist, dist_to_bounds)

        return np.clip(min_dist, 0.03, 4.0)

    def _compute_reward(self, action):
        """
        PHASE 1B: Revised reward function with potential-based shaping.
        Based on Ng et al. (1999) potential-based reward shaping + recent navigation RL literature.
        """
        reward = 0.0

        # Get current state
        dist_to_goal = np.linalg.norm(self.robot_pose[:2] - self.goal_pos)
        min_obstacle_dist = self._compute_mock_scan()

        # === TERMINAL REWARDS ===

        # 1. Goal reached (DOUBLED from +100 to +200)
        if dist_to_goal < self.goal_radius:
            return 200.0  # Dominant positive reward

        # 2. Collision (INCREASED from -50 to -100)
        if min_obstacle_dist < 0.15:
            return -100.0  # Strong penalty for collision

        # === SHAPING REWARDS (Applied every step) ===

        # 3. Potential-based shaping (Ng et al. 1999 - theoretically sound)
        # φ(s) = -λ × distance_to_goal
        # r_potential = γ × φ(s') - φ(s) = γ × (-λ × d') - (-λ × d) = λ × (d - γ × d')
        λ_potential = 10.0  # Scaling factor
        if self._prev_dist_to_goal is not None:
            potential_prev = -λ_potential * self._prev_dist_to_goal
            potential_curr = -λ_potential * dist_to_goal
            r_potential = self.gamma * potential_curr - potential_prev
            reward += r_potential
        self._prev_dist_to_goal = dist_to_goal

        # 4. Heading alignment reward (Xie et al. 2021)
        # Reward moving in the direction of the goal
        goal_direction = self.goal_pos - self.robot_pose[:2]
        goal_angle = np.arctan2(goal_direction[1], goal_direction[0])
        angle_error = self._normalize_angle(goal_angle - self.robot_pose[2])
        heading_alignment = np.cos(angle_error)  # 1.0 if pointing at goal, -1.0 if opposite

        # Only reward when moving (multiply by velocity)
        velocity = action[0]  # throttle [0.0, 1.0]
        λ_heading = 1.0
        r_heading = λ_heading * heading_alignment * velocity
        reward += r_heading

        # 5. Velocity reward (encourage forward motion, overcome conservatism)
        λ_velocity = 0.5
        r_velocity = λ_velocity * velocity
        reward += r_velocity

        # 6. Smoothness penalty (discourage erratic steering) - REDUCED from -0.1 to -0.05
        if self._prev_action is not None:
            action_diff = action - self._prev_action
            r_smoothness = -0.05 * np.linalg.norm(action_diff)
            reward += r_smoothness
        self._prev_action = action.copy()

        # 7. Proximity shaping (soft collision avoidance)
        if min_obstacle_dist < 0.3:  # Danger zone
            r_proximity = -1.0 * (0.3 - min_obstacle_dist)  # Linear penalty
            reward += r_proximity

        # NOTE: REMOVED step penalty (-0.01) - was causing timeout behavior

        return float(reward)

    def _normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        return np.arctan2(np.sin(angle), np.cos(angle))

    def _check_termination(self):
        """Check if episode should terminate"""
        # Goal reached
        dist_to_goal = np.linalg.norm(self.robot_pose[:2] - self.goal_pos)
        if dist_to_goal < self.goal_radius:
            print(f"[Episode {self.episode_count}] GOAL REACHED at step {self.step_count}!")
            return True

        # Collision
        min_dist = self._compute_mock_scan()
        if min_dist < 0.15:
            print(f"[Episode {self.episode_count}] COLLISION at step {self.step_count}")
            return True

        # Out of bounds
        if abs(self.robot_pose[0]) > 5.0 or abs(self.robot_pose[1]) > 5.0:
            print(f"[Episode {self.episode_count}] OUT OF BOUNDS at step {self.step_count}")
            return True

        return False

    def _get_info(self):
        """Get additional info dict"""
        return {
            'step': self.step_count,
            'robot_pos': self.robot_pose[:2].copy(),
            'goal_pos': self.goal_pos.copy(),
            'dist_to_goal': np.linalg.norm(self.robot_pose[:2] - self.goal_pos),
            'min_obstacle_dist': self._compute_mock_scan()
        }

    def close(self):
        """Clean up resources"""
        if self.mode == 'ros':
            # Shutdown ROS node if initialized
            pass
