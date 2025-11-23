"""
Evaluation Script for Trained PPO Models
Tests trained models and collects performance metrics

Usage:
    python evaluate.py --model /path/to/model.zip --episodes 100
"""
import argparse
import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from factory_nav_rl.env_factory_nav import FactoryNavEnv
from tqdm import tqdm


def evaluate_model(model_path, n_episodes=100, mode='mock', render=False):
    """
    Evaluate a trained model

    Args:
        model_path: Path to saved model (.zip file)
        n_episodes: Number of episodes to run
        mode: 'mock' or 'ros'
        render: Whether to render (not implemented)

    Returns:
        dict: Dictionary with evaluation metrics
    """
    print("=" * 60)
    print("PPO Model Evaluation")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Episodes: {n_episodes}")
    print(f"Mode: {mode}")
    print("=" * 60)

    # Load model
    print("\n[1/3] Loading model...")
    model = PPO.load(model_path)
    print(" Model loaded successfully")

    # Create environment
    print("\n[2/3] Creating environment...")
    env = FactoryNavEnv(mode=mode, max_steps=1000)  # Fixed: match training max_steps
    print(" Environment created")

    # Run evaluation
    print(f"\n[3/3] Running {n_episodes} evaluation episodes...")

    episode_rewards = []
    episode_lengths = []
    successes = 0
    collisions = 0
    timeouts = 0
    out_of_bounds = 0

    distances_to_goal = []
    min_obstacle_distances = []

    for episode in tqdm(range(n_episodes), desc="Evaluating"):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        episode_distances = []
        episode_min_dists = []

        while not done:
            # Get action from model (deterministic for evaluation)
            action, _states = model.predict(obs, deterministic=True)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            episode_length += 1
            done = terminated or truncated

            # Track metrics
            episode_distances.append(info['dist_to_goal'])
            episode_min_dists.append(info['min_obstacle_dist'])

        # Record episode results
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        distances_to_goal.append(np.mean(episode_distances))
        min_obstacle_distances.append(np.mean(episode_min_dists))

        # Determine outcome
        final_dist = info['dist_to_goal']
        final_obstacle_dist = info['min_obstacle_dist']

        if final_dist < env.goal_radius:
            successes += 1
        elif final_obstacle_dist < 0.15:
            collisions += 1
        elif episode_length >= env.max_steps:
            timeouts += 1
        elif abs(env.robot_pose[0]) > 5.0 or abs(env.robot_pose[1]) > 5.0:
            out_of_bounds += 1

    env.close()

    # Compute statistics
    results = {
        'n_episodes': n_episodes,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'success_rate': successes / n_episodes * 100,
        'collision_rate': collisions / n_episodes * 100,
        'timeout_rate': timeouts / n_episodes * 100,
        'out_of_bounds_rate': out_of_bounds / n_episodes * 100,
        'mean_dist_to_goal': np.mean(distances_to_goal),
        'mean_min_obstacle_dist': np.mean(min_obstacle_distances),
        'successes': successes,
        'collisions': collisions,
        'timeouts': timeouts,
        'out_of_bounds': out_of_bounds
    }

    return results, episode_rewards, episode_lengths


def print_results(results):
    """Print evaluation results in a nice format"""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Episodes: {results['n_episodes']}")
    print(f"\nReward Statistics:")
    print(f"  Mean: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"\nEpisode Length:")
    print(f"  Mean: {results['mean_length']:.1f} ± {results['std_length']:.1f} steps")
    print(f"\nOutcome Rates:")
    print(f"  Success:       {results['success_rate']:6.2f}% ({results['successes']} episodes)")
    print(f"  Collision:     {results['collision_rate']:6.2f}% ({results['collisions']} episodes)")
    print(f"  Timeout:       {results['timeout_rate']:6.2f}% ({results['timeouts']} episodes)")
    print(f"  Out of Bounds: {results['out_of_bounds_rate']:6.2f}% ({results['out_of_bounds']} episodes)")
    print(f"\nNavigation Metrics:")
    print(f"  Avg distance to goal: {results['mean_dist_to_goal']:.2f}m")
    print(f"  Avg min obstacle dist: {results['mean_min_obstacle_dist']:.2f}m")
    print("=" * 60)


def save_results(results, episode_rewards, episode_lengths, output_dir):
    """Save results to CSV files"""
    os.makedirs(output_dir, exist_ok=True)

    # Save summary
    summary_df = pd.DataFrame([results])
    summary_path = os.path.join(output_dir, 'evaluation_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\n Summary saved to: {summary_path}")

    # Save episode data
    episodes_df = pd.DataFrame({
        'episode': range(len(episode_rewards)),
        'reward': episode_rewards,
        'length': episode_lengths
    })
    episodes_path = os.path.join(output_dir, 'evaluation_episodes.csv')
    episodes_df.to_csv(episodes_path, index=False)
    print(f" Episode data saved to: {episodes_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained PPO model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model (.zip file)')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of evaluation episodes (default: 100)')
    parser.add_argument('--mode', type=str, default='mock', choices=['mock', 'ros'],
                        help='Environment mode (default: mock)')
    parser.add_argument('--output', type=str, default='./evaluation_results',
                        help='Output directory for results (default: ./evaluation_results)')

    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model):
        print(f"❌ Error: Model file not found: {args.model}")
        return

    # Run evaluation
    results, episode_rewards, episode_lengths = evaluate_model(
        args.model,
        n_episodes=args.episodes,
        mode=args.mode
    )

    # Print results
    print_results(results)

    # Save results
    save_results(results, episode_rewards, episode_lengths, args.output)

    print(f"\n Evaluation complete!")


if __name__ == "__main__":
    main()
