#!/usr/bin/env python3
"""
Real-time visualization of trained PPO agent navigating the factory environment.
Perfect for demonstrations to show the agent in action!

Usage:
    python3 visualize_demo.py --model ~/training_extended_1M/ppo_factory_nav_final.zip
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from stable_baselines3 import PPO
from env_factory_nav import FactoryNavEnv


class FactoryNavVisualizer:
    def __init__(self, model_path, curriculum_stage='full', max_episodes=5):
        """Initialize visualizer with trained model"""
        print(f"Loading model from {model_path}...")
        self.model = PPO.load(model_path)

        print(f"Creating environment (curriculum: {curriculum_stage})...")
        self.env = FactoryNavEnv(mode='mock', curriculum_stage=curriculum_stage, max_steps=500)

        self.max_episodes = max_episodes
        self.current_episode = 0
        self.episode_reward = 0
        self.episode_steps = 0

        # Trajectory tracking
        self.trajectory_x = []
        self.trajectory_y = []

        # Setup plot
        self.setup_plot()

        # Reset environment
        self.obs, _ = self.env.reset()

    def setup_plot(self):
        """Setup matplotlib figure with subplots"""
        self.fig = plt.figure(figsize=(16, 9))

        # Main environment view (larger)
        self.ax_env = plt.subplot(2, 3, (1, 4))
        self.ax_env.set_xlim(-6, 6)
        self.ax_env.set_ylim(-6, 6)
        self.ax_env.set_aspect('equal')
        self.ax_env.grid(True, alpha=0.3)
        self.ax_env.set_title('Factory Navigation Environment', fontsize=14, fontweight='bold')
        self.ax_env.set_xlabel('X position (m)')
        self.ax_env.set_ylabel('Y position (m)')

        # Camera view
        self.ax_camera = plt.subplot(2, 3, 2)
        self.ax_camera.set_title('Camera Feed', fontsize=12)
        self.ax_camera.axis('off')

        # Lidar scan
        self.ax_lidar = plt.subplot(2, 3, 3)
        self.ax_lidar.set_title('Lidar Scan', fontsize=12)
        self.ax_lidar.set_xlabel('Time step')
        self.ax_lidar.set_ylabel('Min distance (m)')
        self.lidar_history = []

        # Metrics
        self.ax_metrics = plt.subplot(2, 3, 5)
        self.ax_metrics.axis('off')
        self.ax_metrics.set_title('Episode Metrics', fontsize=12)

        # Reward history
        self.ax_reward = plt.subplot(2, 3, 6)
        self.ax_reward.set_title('Reward History', fontsize=12)
        self.ax_reward.set_xlabel('Time step')
        self.ax_reward.set_ylabel('Reward')
        self.reward_history = []

        plt.tight_layout()

    def draw_environment(self):
        """Draw the environment state"""
        self.ax_env.clear()
        self.ax_env.set_xlim(-6, 6)
        self.ax_env.set_ylim(-6, 6)
        self.ax_env.set_aspect('equal')
        self.ax_env.grid(True, alpha=0.3)
        self.ax_env.set_title(f'Episode {self.current_episode + 1}/{self.max_episodes} | Step {self.episode_steps}',
                              fontsize=14, fontweight='bold')
        self.ax_env.set_xlabel('X position (m)')
        self.ax_env.set_ylabel('Y position (m)')

        # Draw boundaries
        boundary = patches.Rectangle((-5, -5), 10, 10, linewidth=2,
                                     edgecolor='black', facecolor='none', linestyle='--')
        self.ax_env.add_patch(boundary)

        # Draw obstacles
        for obs in self.env.obstacles:
            obs_x, obs_y, obs_r = obs
            circle = patches.Circle((obs_x, obs_y), obs_r, color='red', alpha=0.6)
            self.ax_env.add_patch(circle)
            self.ax_env.text(obs_x, obs_y, 'OBS', ha='center', va='center',
                           fontsize=8, fontweight='bold', color='white')

        # Draw goal
        goal_circle = patches.Circle(self.env.goal_pos, self.env.goal_radius,
                                     color='green', alpha=0.4)
        self.ax_env.add_patch(goal_circle)
        self.ax_env.plot(self.env.goal_pos[0], self.env.goal_pos[1], 'g*',
                        markersize=20, label='Goal')

        # Draw trajectory
        if len(self.trajectory_x) > 1:
            self.ax_env.plot(self.trajectory_x, self.trajectory_y, 'b-',
                           alpha=0.5, linewidth=2, label='Trajectory')

        # Draw robot
        robot_x, robot_y, robot_theta = self.env.robot_pose

        # Robot body (circle)
        robot_circle = patches.Circle((robot_x, robot_y), 0.15,
                                     color='blue', alpha=0.8)
        self.ax_env.add_patch(robot_circle)

        # Robot heading (arrow)
        arrow_len = 0.3
        dx = arrow_len * np.cos(robot_theta)
        dy = arrow_len * np.sin(robot_theta)
        self.ax_env.arrow(robot_x, robot_y, dx, dy,
                         head_width=0.15, head_length=0.1,
                         fc='darkblue', ec='darkblue', linewidth=2)

        # Distance to goal line
        self.ax_env.plot([robot_x, self.env.goal_pos[0]],
                        [robot_y, self.env.goal_pos[1]],
                        'g--', alpha=0.3, linewidth=1)

        # Add legend
        self.ax_env.legend(loc='upper right', fontsize=10)

    def update_camera_view(self):
        """Update camera image display"""
        self.ax_camera.clear()
        self.ax_camera.set_title('Camera Feed', fontsize=12)
        self.ax_camera.axis('off')

        # Display camera image
        img = self.obs['image']
        self.ax_camera.imshow(img)

    def update_lidar_plot(self):
        """Update lidar scan history"""
        self.ax_lidar.clear()
        self.ax_lidar.set_title('Lidar Scan (Min Distance)', fontsize=12)
        self.ax_lidar.set_xlabel('Time step')
        self.ax_lidar.set_ylabel('Distance (m)')

        if len(self.lidar_history) > 0:
            self.ax_lidar.plot(self.lidar_history, 'r-', linewidth=2)
            self.ax_lidar.axhline(y=0.3, color='orange', linestyle='--',
                                 label='Danger zone', alpha=0.7)
            self.ax_lidar.axhline(y=0.15, color='red', linestyle='--',
                                 label='Collision', alpha=0.7)
            self.ax_lidar.legend(loc='upper right', fontsize=8)
            self.ax_lidar.set_ylim(0, 4)

    def update_metrics(self):
        """Update metrics display"""
        self.ax_metrics.clear()
        self.ax_metrics.axis('off')

        # Get current metrics
        dist_to_goal = np.linalg.norm(self.env.robot_pose[:2] - self.env.goal_pos)
        min_obstacle_dist = self.env._compute_mock_scan()
        velocity = self.env.robot_velocity

        # Format metrics text
        metrics_text = f"""
Episode: {self.current_episode + 1}/{self.max_episodes}
Step: {self.episode_steps}
Episode Reward: {self.episode_reward:.2f}

Distance to Goal: {dist_to_goal:.2f} m
Min Obstacle Dist: {min_obstacle_dist:.2f} m

Robot Velocity:
  Linear: {velocity[0]:.2f} m/s
  Angular: {velocity[1]:.2f} rad/s

Robot Pose:
  X: {self.env.robot_pose[0]:.2f} m
  Y: {self.env.robot_pose[1]:.2f} m
  Θ: {np.degrees(self.env.robot_pose[2]):.1f}°
"""

        self.ax_metrics.text(0.1, 0.9, metrics_text, transform=self.ax_metrics.transAxes,
                           fontsize=11, verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    def update_reward_plot(self):
        """Update reward history"""
        self.ax_reward.clear()
        self.ax_reward.set_title('Step Reward History', fontsize=12)
        self.ax_reward.set_xlabel('Time step')
        self.ax_reward.set_ylabel('Reward')

        if len(self.reward_history) > 0:
            self.ax_reward.plot(self.reward_history, 'g-', linewidth=2)
            self.ax_reward.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    def animate(self, frame):
        """Animation update function"""
        # Get action from trained policy
        action, _ = self.model.predict(self.obs, deterministic=True)

        # Step environment
        self.obs, reward, terminated, truncated, info = self.env.step(action)

        # Update tracking
        self.episode_reward += reward
        self.episode_steps += 1
        self.trajectory_x.append(self.env.robot_pose[0])
        self.trajectory_y.append(self.env.robot_pose[1])
        self.lidar_history.append(self.obs['scan'][0])
        self.reward_history.append(reward)

        # Update all plots
        self.draw_environment()
        self.update_camera_view()
        self.update_lidar_plot()
        self.update_metrics()
        self.update_reward_plot()

        # Check episode end
        if terminated or truncated:
            print(f"\nEpisode {self.current_episode + 1} finished:")
            print(f"  Steps: {self.episode_steps}")
            print(f"  Reward: {self.episode_reward:.2f}")
            print(f"  Outcome: {'SUCCESS' if terminated and info['dist_to_goal'] < self.env.goal_radius else 'TIMEOUT/COLLISION'}")

            self.current_episode += 1

            if self.current_episode >= self.max_episodes:
                print("\nDemo completed! Closing visualization...")
                plt.close()
                return

            # Reset for next episode
            self.obs, _ = self.env.reset()
            self.episode_reward = 0
            self.episode_steps = 0
            self.trajectory_x = []
            self.trajectory_y = []
            self.lidar_history = []
            self.reward_history = []

    def run(self):
        """Run the visualization"""
        print("\nStarting real-time demonstration...")
        print("Close the window to stop early.\n")

        # Create animation
        anim = FuncAnimation(self.fig, self.animate, interval=100, cache_frame_data=False)
        plt.show()

        print("\nDemonstration completed!")


def main():
    parser = argparse.ArgumentParser(description='Visualize trained PPO agent')
    parser.add_argument('--model', type=str,
                       default='/home/picar/training_extended_1M/ppo_factory_nav_final.zip',
                       help='Path to trained model')
    parser.add_argument('--curriculum', type=str, default='full',
                       choices=['simple', 'medium', 'full'],
                       help='Curriculum stage (simple=no obstacles, medium=1 obstacle, full=3 obstacles)')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of episodes to run')

    args = parser.parse_args()

    print("=" * 60)
    print("Factory Navigation RL - Live Demonstration")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Curriculum: {args.curriculum}")
    print(f"Episodes: {args.episodes}")
    print("=" * 60)

    # Create and run visualizer
    viz = FactoryNavVisualizer(
        model_path=args.model,
        curriculum_stage=args.curriculum,
        max_episodes=args.episodes
    )
    viz.run()


if __name__ == '__main__':
    main()
