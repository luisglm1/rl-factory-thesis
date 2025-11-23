"""
Curriculum Learning Training Script for Factory Navigation
Trains PPO agent in 3 stages: simple → medium → full complexity

Stage 1 (Simple):  2-3m goal, no obstacles (200k timesteps)
Stage 2 (Medium):  4-6m goal, 1 obstacle (300k timesteps)
Stage 3 (Full):    3.5-4.5m goal, 3 obstacles (500k timesteps)

Usage:
    python train_curriculum.py --stage all
    python train_curriculum.py --stage simple --timesteps 200000
    python train_curriculum.py --stage medium --resume path/to/simple_model.zip
"""
import argparse
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from factory_nav_rl.env_factory_nav import FactoryNavEnv


def make_env(mode='mock', curriculum_stage='full', max_steps=1000, gamma=0.995):
    """Create a factory navigation environment with curriculum stage"""
    def _init():
        return FactoryNavEnv(
            mode=mode,
            max_steps=max_steps,
            gamma=gamma,
            curriculum_stage=curriculum_stage
        )
    return _init


def train_stage(stage_name, stage_config, args, resume_model=None):
    """
    Train a single curriculum stage.

    Args:
        stage_name: 'simple', 'medium', or 'full'
        stage_config: dict with 'timesteps', 'max_steps', 'n_envs'
        args: command line arguments
        resume_model: path to model to load weights from (for transfer learning)

    Returns:
        final_model_path: path to the saved final model
    """
    print("\n" + "=" * 70)
    print(f"CURRICULUM STAGE: {stage_name.upper()}")
    print("=" * 70)
    print(f"Timesteps: {stage_config['timesteps']:,}")
    print(f"Max steps per episode: {stage_config['max_steps']}")
    print(f"Parallel environments: {stage_config['n_envs']}")
    if resume_model:
        print(f"Resuming from: {resume_model}")
    print("=" * 70)

    # Create log directory for this stage
    log_dir = os.path.join(args.log_dir, f"stage_{stage_name}")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'tensorboard'), exist_ok=True)

    # Create vectorized environment
    print(f"\n[1/4] Creating {stage_name} curriculum environments...")
    env = DummyVecEnv([
        make_env(
            mode=args.mode,
            curriculum_stage=stage_name,
            max_steps=stage_config['max_steps'],
            gamma=0.995
        ) for _ in range(stage_config['n_envs'])
    ])
    env = VecMonitor(env, filename=os.path.join(log_dir, "monitor"))
    print(f" Created {stage_config['n_envs']} parallel environments ({stage_name} difficulty)")

    # Create or load PPO model
    print(f"\n[2/4] Initializing PPO model...")
    if resume_model and os.path.exists(resume_model):
        print(f"   Loading weights from: {resume_model}")
        model = PPO.load(resume_model, env=env)
        # Update hyperparameters (they might have changed)
        model.gamma = 0.995
        model.ent_coef = 0.01
        model.tensorboard_log = os.path.join(log_dir, "tensorboard")
        print(" Model loaded with transferred weights")
    else:
        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=os.path.join(log_dir, "tensorboard"),
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256])
            )
        )
        print(" New PPO model initialized")

    print(f"   Hyperparameters:")
    print(f"   - Discount factor (gamma): 0.995")
    print(f"   - Entropy coefficient: 0.01")
    print(f"   - Learning rate: 3e-4")
    print(f"   - Network: [256, 256]")

    # Setup callbacks
    print(f"\n[3/4] Setting up callbacks...")
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10000, stage_config['timesteps'] // 10),
        save_path=os.path.join(log_dir, 'checkpoints'),
        name_prefix=f'ppo_{stage_name}'
    )
    print(f" Checkpoints will be saved to: {log_dir}/checkpoints/")

    # Start training
    print(f"\n[4/4] Starting {stage_name} training...")
    print("-" * 70)
    try:
        model.learn(
            total_timesteps=stage_config['timesteps'],
            callback=checkpoint_callback,
            progress_bar=True,
            reset_num_timesteps=False  # Continue timestep counting for transfer learning
        )
        print("-" * 70)
        print(f"\n {stage_name.upper()} stage completed successfully!")

        # Save final model
        final_model_path = os.path.join(log_dir, f"ppo_{stage_name}_final")
        model.save(final_model_path)
        print(f" Saved final model to: {final_model_path}.zip")

        return final_model_path + ".zip"

    except KeyboardInterrupt:
        print(f"\n\n⚠️  {stage_name} training interrupted by user!")
        interrupted_model_path = os.path.join(log_dir, f"ppo_{stage_name}_interrupted")
        model.save(interrupted_model_path)
        print(f" Saved interrupted model to: {interrupted_model_path}.zip")
        return interrupted_model_path + ".zip"

    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description='Curriculum training for factory navigation')
    parser.add_argument('--stage', type=str, default='all',
                        choices=['simple', 'medium', 'full', 'all'],
                        help='Which curriculum stage to train (default: all)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint (for single stage training)')
    parser.add_argument('--mode', type=str, default='mock', choices=['mock', 'ros'],
                        help='Environment mode: mock or ros (default: mock)')
    parser.add_argument('--log_dir', type=str, default='./logs_curriculum',
                        help='Base directory for logs (default: ./logs_curriculum)')
    parser.add_argument('--n_envs', type=int, default=8,
                        help='Number of parallel environments (default: 8)')

    args = parser.parse_args()

    # Curriculum configuration
    curriculum = {
        'simple': {
            'timesteps': 200000,
            'max_steps': 500,   # Easier task, shorter episodes OK
            'n_envs': args.n_envs
        },
        'medium': {
            'timesteps': 300000,
            'max_steps': 750,   # Medium difficulty
            'n_envs': args.n_envs
        },
        'full': {
            'timesteps': 500000,
            'max_steps': 1000,  # Full complexity, longer episodes
            'n_envs': args.n_envs
        }
    }

    print("\n" + "=" * 70)
    print("CURRICULUM LEARNING: FACTORY NAVIGATION")
    print("=" * 70)
    print("This training uses 3-stage curriculum learning:")
    print("  1. SIMPLE:  2-3m goal, no obstacles (200k steps)")
    print("  2. MEDIUM:  4-6m goal, 1 obstacle (300k steps)")
    print("  3. FULL:    3.5-4.5m goal, 3 obstacles (500k steps)")
    print("Total training: 1M timesteps")
    print("=" * 70)

    if args.stage == 'all':
        # Train all stages sequentially with transfer learning
        print("\n Running full curriculum (all 3 stages)...")

        # Stage 1: Simple
        simple_model = train_stage('simple', curriculum['simple'], args)

        # Stage 2: Medium (transfer from simple)
        medium_model = train_stage('medium', curriculum['medium'], args, resume_model=simple_model)

        # Stage 3: Full (transfer from medium)
        full_model = train_stage('full', curriculum['full'], args, resume_model=medium_model)

        print("\n" + "=" * 70)
        print(" CURRICULUM TRAINING COMPLETE!")
        print("=" * 70)
        print(f"Stage 1 (Simple) model: {simple_model}")
        print(f"Stage 2 (Medium) model: {medium_model}")
        print(f"Stage 3 (Full) model: {full_model}")
        print("\nTo evaluate the final model:")
        print(f"  python evaluate.py --model {full_model} --episodes 100")
        print("=" * 70)

    else:
        # Train single stage
        train_stage(args.stage, curriculum[args.stage], args, resume_model=args.resume)


if __name__ == "__main__":
    main()
