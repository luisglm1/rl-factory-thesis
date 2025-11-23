"""
PPO Training Script for Factory Navigation
Trains a PPO agent using stable-baselines3

Usage:
    # Short test run
    python train_ppo.py --timesteps 10000

    # Full training
    python train_ppo.py --timesteps 1000000
"""
import argparse
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from factory_nav_rl.env_factory_nav import FactoryNavEnv


def make_env(mode='mock', max_steps=1000, gamma=0.995):
    """Create a factory navigation environment"""
    def _init():
        return FactoryNavEnv(mode=mode, max_steps=max_steps, gamma=gamma)
    return _init


def main():
    parser = argparse.ArgumentParser(description='Train PPO agent for factory navigation')
    parser.add_argument('--timesteps', type=int, default=10000,
                        help='Total timesteps for training (default: 10000)')
    parser.add_argument('--n_envs', type=int, default=4,
                        help='Number of parallel environments (default: 4)')
    parser.add_argument('--mode', type=str, default='mock', choices=['mock', 'ros'],
                        help='Environment mode: mock or ros (default: mock)')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Directory for logs and checkpoints (default: ./logs)')
    parser.add_argument('--save_freq', type=int, default=10000,
                        help='Save checkpoint every N steps (default: 10000)')

    args = parser.parse_args()

    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(os.path.join(args.log_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(args.log_dir, 'tensorboard'), exist_ok=True)

    print("=" * 60)
    print("PPO Training for Factory Navigation")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Total timesteps: {args.timesteps:,}")
    print(f"Parallel environments: {args.n_envs}")
    print(f"Log directory: {args.log_dir}")
    print("=" * 60)

    # Create vectorized environment
    print("\n[1/4] Creating environments...")
    # PHASE 1: Use updated max_steps=1000 and gamma=0.995
    env = DummyVecEnv([make_env(args.mode, max_steps=1000, gamma=0.995) for _ in range(args.n_envs)])
    env = VecMonitor(env, filename=os.path.join(args.log_dir, "monitor"))
    print(f" Created {args.n_envs} parallel environments (max_steps=1000)")


    # Create PPO model with tuned hyperparameters
    print("\n[2/4] Initializing PPO model...")
    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.995,  # PHASE 1C: Increased from 0.99 to 0.995 for longer horizon
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # PHASE 1C: Entropy regularization for exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=os.path.join(args.log_dir, "tensorboard"),
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        )
    )
    print(" PPO model initialized with PHASE 1 hyperparameters:")
    print(f"   - Learning rate: 3e-4")
    print(f"   - Discount factor (gamma): 0.995 (was 0.99)")
    print(f"   - Entropy coefficient: 0.01 (exploration boost)")
    print(f"   - Steps per update: 2048")
    print(f"   - Batch size: 64")
    print(f"   - Network: [256, 256] for both policy and value")

    # Setup callbacks
    print("\n[3/4] Setting up callbacks...")
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=os.path.join(args.log_dir, 'checkpoints'),
        name_prefix='ppo_factory_nav'
    )
    print(f" Checkpoint callback: save every {args.save_freq:,} steps")

    # Start training
    print("\n[4/4] Starting training...")
    print("-" * 60)
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=checkpoint_callback,
            progress_bar=True
        )
        print("-" * 60)
        print("\n Training completed successfully!")

        # Save final model
        final_model_path = os.path.join(args.log_dir, "ppo_factory_nav_final")
        model.save(final_model_path)
        print(f"Saved final model to: {final_model_path}.zip")

        # Print summary
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        print(f"Total timesteps: {args.timesteps:,}")
        print(f"Model saved to: {final_model_path}.zip")
        print(f"Logs saved to: {args.log_dir}/")
        print(f"TensorBoard logs: {os.path.join(args.log_dir, 'tensorboard')}")
        print("\nTo view training progress:")
        print(f"  tensorboard --logdir {os.path.join(args.log_dir, 'tensorboard')}")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user!")
        interrupted_model_path = os.path.join(args.log_dir, "ppo_factory_nav_interrupted")
        model.save(interrupted_model_path)
        print(f" Saved interrupted model to: {interrupted_model_path}.zip")

    finally:
        env.close()


if __name__ == "__main__":
    main()
