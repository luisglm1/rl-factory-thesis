#!/usr/bin/env python3
"""
Generate Thesis Figures for Stage 3 Only
=========================================

Generates:
- Training learning curve (reward progression)
- Episode length progression
- PPO training diagnostics (Total Loss, Policy Gradient Loss, Value Loss, Entropy Loss)

"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing import event_accumulator

# Set publication-quality defaults for IEEE format
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif'
})
sns.set_style("whitegrid")


def load_tensorboard_data(logdir):
    """Load data from TensorBoard logs"""
    print(f"Loading TensorBoard data from: {logdir}")

    # Find the event file
    event_file = None
    for root, dirs, files in os.walk(logdir):
        for file in files:
            if file.startswith('events.out.tfevents'):
                event_file = os.path.join(root, file)
                break
        if event_file:
            break

    if not event_file:
        print(f"No TensorBoard event file found in {logdir}")
        return None

    print(f"Found event file: {event_file}")

    # Load events
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()

    data = {}

    # Extract scalar data
    tags = ea.Tags()['scalars']
    print(f"Found {len(tags)} scalar metrics: {tags}")

    for tag in tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        data[tag] = pd.DataFrame({'step': steps, 'value': values})

    return data


def generate_figure_5(data, output_dir):
    """
    Generate Figure 5: Training Learning Curve (Reward) - Stage 3 only
    """
    print("\n[Figure 5] Generating training learning curve (reward)...")

    fig, ax = plt.subplots(figsize=(10, 6))

    if 'rollout/ep_rew_mean' in data:
        df = data['rollout/ep_rew_mean'].copy()
        df = df.sort_values('step').reset_index(drop=True)

        # Smooth curve using rolling average
        window = min(20, max(1, len(df) // 10))
        if window > 1 and len(df) > window:
            df['smoothed'] = df['value'].rolling(window=window, center=True).mean()
        else:
            df['smoothed'] = df['value']

        # Plot raw data with transparency
        ax.plot(df['step'], df['value'], alpha=0.3, color='steelblue',
                label='Raw', linewidth=0.8)

        # Plot smoothed curve
        ax.plot(df['step'], df['smoothed'], linewidth=2.5, color='darkblue',
                label='Smoothed')

        ax.set_xlabel('Training Steps', fontweight='bold')
        ax.set_ylabel('Episode Reward', fontweight='bold')
        ax.set_title('PPO Training Progress: Episode Reward', fontweight='bold', pad=15)

        # Add final reward annotation
        final_reward = df['value'].iloc[-1]
        ax.axhline(y=final_reward, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax.text(df['step'].max() * 0.7, final_reward, f'Final: {final_reward:.1f}',
                verticalalignment='bottom', fontsize=9)

        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        # Format x-axis in scientific notation
        ax.ticklabel_format(style='scientific', axis='x', scilimits=(6,6))

        filepath = os.path.join(output_dir, 'learning_curve_reward.png')
        plt.savefig(filepath)
        plt.close()
        print(f"  Saved: {filepath}")
        return True
    else:
        print("  Warning: No reward data found (rollout/ep_rew_mean)")
        plt.close()
        return False


def generate_figure_6(data, output_dir):
    """
    Generate Figure 6: Episode Length Progression - Stage 3 only
    """
    print("\n[Figure 6] Generating episode length progression...")

    fig, ax = plt.subplots(figsize=(10, 6))

    if 'rollout/ep_len_mean' in data:
        df = data['rollout/ep_len_mean'].copy()
        df = df.sort_values('step').reset_index(drop=True)

        # Smooth curve
        window = min(20, max(1, len(df) // 10))
        if window > 1 and len(df) > window:
            df['smoothed'] = df['value'].rolling(window=window, center=True).mean()
        else:
            df['smoothed'] = df['value']

        # Plot raw data
        ax.plot(df['step'], df['value'], alpha=0.3, color='coral',
                label='Raw', linewidth=0.8)

        # Plot smoothed curve
        ax.plot(df['step'], df['smoothed'], linewidth=2.5, color='darkred',
                label='Smoothed')

        ax.set_xlabel('Training Steps', fontweight='bold')
        ax.set_ylabel('Episode Length (steps)', fontweight='bold')
        ax.set_title('PPO Training Progress: Episode Length', fontweight='bold', pad=15)

        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        # Format x-axis in scientific notation
        ax.ticklabel_format(style='scientific', axis='x', scilimits=(6,6))

        filepath = os.path.join(output_dir, 'learning_curve_length.png')
        plt.savefig(filepath)
        plt.close()
        print(f"  Saved: {filepath}")
        return True
    else:
        print("  Warning: No episode length data found (rollout/ep_len_mean)")
        plt.close()
        return False


def generate_figure_7(data, output_dir):
    """
    Generate Figure 7: PPO Training Diagnostics (matching thesis PDF format)

    4 subplots:
    - Total Loss over Training
    - Policy Gradient Loss over Training
    - Value Loss over Training
    - Entropy Loss over Training
    """
    print("\n[Figure 7] Generating PPO training diagnostics...")

    # Define metrics to plot - matching thesis Figure 7 exactly
    # Note: entropy_loss in SB3 is -ent_coef * entropy, so we negate it to show actual entropy
    metrics_config = [
        ('train/loss', 'Total Loss over Training', 'purple', False),
        ('train/policy_gradient_loss', 'Policy Gradient Loss over Training', 'green', False),
        ('train/value_loss', 'Value Loss over Training', 'orange', False),
        ('train/entropy_loss', 'Entropy over Training', 'brown', True)  # negate=True
    ]

    # Check which metrics are available
    available_metrics = []
    for tag, name, color, negate in metrics_config:
        if tag in data:
            available_metrics.append((tag, name, color, negate))
        else:
            print(f"  Warning: Metric {tag} not found")

    if len(available_metrics) < 2:
        print("  Error: Not enough metrics available for Figure 7")
        return False

    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (tag, name, color, negate) in enumerate(available_metrics[:4]):
        ax = axes[idx]
        df = data[tag].copy()
        df = df.sort_values('step').reset_index(drop=True)

        # Get values, negating if needed (for entropy_loss -> entropy)
        values = -df['value'] if negate else df['value']

        # Plot the data
        ax.plot(df['step'], values, color=color, linewidth=1.5, alpha=0.8)

        ax.set_xlabel('Training Steps')
        ax.set_ylabel(name.split(' over ')[0])
        ax.set_title(name, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Format x-axis in scientific notation
        ax.ticklabel_format(style='scientific', axis='x', scilimits=(6,6))

    # Hide unused subplots
    for idx in range(len(available_metrics), 4):
        axes[idx].axis('off')

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'training_metrics.png')
    plt.savefig(filepath)
    plt.close()
    print(f"  Saved: {filepath}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Generate Thesis Figures for Stage 3 Only',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate figures from Stage 3 training
  python generate_stage3_figures.py \\
      --tensorboard ~/training_curriculum_1M/stage_full/tensorboard \\
      --output ~/Documents/rl-factory-thesis-skeleton/figures/curriculum
        """
    )
    parser.add_argument('--tensorboard', type=str, required=True,
                        help='Path to TensorBoard logs directory')
    parser.add_argument('--output', type=str, default='./figures',
                        help='Output directory for figures (default: ./figures)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("Thesis Figure Generation - Stage 3 Only")
    print("=" * 60)
    print(f"TensorBoard logs: {args.tensorboard}")
    print(f"Output directory: {args.output}")
    print("=" * 60)

    # Check if tensorboard directory exists
    if not os.path.exists(args.tensorboard):
        print(f"\nError: TensorBoard directory not found: {args.tensorboard}")
        return 1

    # Load data
    data = load_tensorboard_data(args.tensorboard)
    if data is None:
        print("\nFailed to load TensorBoard data")
        return 1

    # Generate figures
    results = {
        'Figure 5 (learning_curve_reward.png)': generate_figure_5(data, args.output),
        'Figure 6 (learning_curve_length.png)': generate_figure_6(data, args.output),
        'Figure 7 (training_metrics.png)': generate_figure_7(data, args.output)
    }

    # Summary
    print("\n" + "=" * 60)
    print("Generation Summary")
    print("=" * 60)

    all_success = True
    for fig_name, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"  {fig_name}: {status}")
        if not success:
            all_success = False

    print("=" * 60)

    if all_success:
        print(f"\nAll figures saved to: {args.output}")
        print("\nGenerated files:")
        print(f"  - learning_curve_reward.png  (Figure 5)")
        print(f"  - learning_curve_length.png  (Figure 6)")
        print(f"  - training_metrics.png       (Figure 7)")
    else:
        print("\nSome figures failed to generate. Check TensorBoard logs.")

    return 0 if all_success else 1


if __name__ == "__main__":
    exit(main())
