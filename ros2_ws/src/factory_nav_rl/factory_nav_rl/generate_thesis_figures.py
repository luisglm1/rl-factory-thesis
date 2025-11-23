#!/usr/bin/env python3
"""
This script generates the following figures
- Training learning curve (reward progression) - ALL 3 CURRICULUM STAGES
- Episode length progression - ALL 3 CURRICULUM STAGES
- PPO training diagnostics (policy loss, value loss, entropy, explained variance)

"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing import event_accumulator
from matplotlib.patches import Patch

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


def load_all_curriculum_stages(base_dir):
    """Load and combine data from all 3 curriculum stages"""
    stages = [
        ('Stage 1: Empty', 'stage_simple', '#3498db', 200000),
        ('Stage 2: Sparse', 'stage_medium', '#f39c12', 300000),
        ('Stage 3: Full Factory', 'stage_full', '#27ae60', 500000)
    ]

    all_data = {}
    stage_info = []
    cumulative_steps = 0

    for stage_name, stage_dir, color, duration in stages:
        stage_path = os.path.join(base_dir, stage_dir, 'tensorboard')

        if not os.path.exists(stage_path):
            print(f"  Warning: {stage_path} not found")
            continue

        data = load_tensorboard_data(stage_path)
        if data is None:
            continue

        # Record stage boundaries for plotting
        if 'rollout/ep_rew_mean' in data:
            df = data['rollout/ep_rew_mean']
            original_min_step = df['step'].min()
            original_max_step = df['step'].max()

            # Normalize steps to be continuous (0 to 1M)
            stage_duration = original_max_step - original_min_step

            # Scale to target duration and offset
            for tag in data:
                if len(data[tag]) > 0:
                    orig_steps = data[tag]['step'].values
                    # Normalize to 0-1, then scale to duration and add offset
                    normalized = (orig_steps - original_min_step) / max(1, stage_duration)
                    data[tag]['step'] = cumulative_steps + normalized * duration

            stage_info.append({
                'name': stage_name,
                'start': cumulative_steps,
                'end': cumulative_steps + duration,
                'color': color,
                'original_start': original_min_step,
                'original_end': original_max_step
            })

            cumulative_steps += duration

        # Merge into all_data
        for tag in data:
            if tag not in all_data:
                all_data[tag] = data[tag].copy()
            else:
                all_data[tag] = pd.concat([all_data[tag], data[tag]], ignore_index=True)

    return all_data, stage_info


def generate_figure_5(data, output_dir, stage_info=None):
    """
    Generate Figure 5: Training Learning Curve (Reward)

    Shows the episode reward progression across ALL 3 curriculum stages.
    """
    print("\n[Figure 5] Generating training learning curve (reward)...")

    fig, ax = plt.subplots(figsize=(12, 6))

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

        # Add confidence band using rolling std
        if len(df) > window:
            df['std'] = df['value'].rolling(window=window, center=True).std()
            ax.fill_between(df['step'],
                           df['smoothed'] - df['std'],
                           df['smoothed'] + df['std'],
                           alpha=0.15, color='steelblue')

        # Add stage boundaries and shading
        if stage_info:
            for i, stage in enumerate(stage_info):
                # Vertical line at stage boundary
                if i > 0:
                    ax.axvline(x=stage['start'], color='gray', linestyle='--',
                              linewidth=1.5, alpha=0.7)

                # Stage label at top
                mid_x = (stage['start'] + stage['end']) / 2
                ax.text(mid_x, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 350,
                       stage['name'].split(':')[0],
                       ha='center', va='top', fontsize=10, fontweight='bold',
                       color=stage['color'],
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        ax.set_xlabel('Training Steps', fontweight='bold')
        ax.set_ylabel('Mean Episode Reward', fontweight='bold')
        ax.set_title('Curriculum Learning: Episode Reward Progression Across All Stages',
                    fontweight='bold', pad=15)

        # Format x-axis
        ax.set_xlim([0, 1000000])
        ax.set_xticks([0, 200000, 500000, 1000000])
        ax.set_xticklabels(['0', '200k', '500k', '1M'])

        # Add annotations for initial and final values
        initial_reward = df['value'].iloc[:5].mean()
        final_reward = df['value'].iloc[-5:].mean()

        ax.axhline(y=initial_reward, color='red', linestyle=':', alpha=0.5, linewidth=1)
        ax.axhline(y=final_reward, color='green', linestyle='--', alpha=0.5, linewidth=1)

        # Annotate with improvement
        improvement = final_reward - initial_reward
        ax.text(50000, initial_reward + 10, f'Initial: {initial_reward:.0f}',
                fontsize=9, color='red', fontweight='bold')
        ax.text(800000, final_reward + 10, f'Final: {final_reward:.0f}',
                fontsize=9, color='green', fontweight='bold')

        # Add improvement annotation
        ax.annotate(f'+{improvement:.0f} reward\n({improvement/max(1,abs(initial_reward))*100:.0f}% improvement)',
                   xy=(500000, (initial_reward + final_reward)/2),
                   fontsize=10, ha='center',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))

        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        filepath = os.path.join(output_dir, 'learning_curve_reward.png')
        plt.savefig(filepath)
        plt.close()
        print(f"  Saved: {filepath}")
        return True
    else:
        print("  Warning: No reward data found (rollout/ep_rew_mean)")
        plt.close()
        return False


def generate_figure_6(data, output_dir, stage_info=None):
    """
    Generate Figure 6: Episode Length Progression

    Shows how episode length evolves across ALL 3 curriculum stages.
    """
    print("\n[Figure 6] Generating episode length progression...")

    fig, ax = plt.subplots(figsize=(12, 6))

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

        # Add confidence band
        if len(df) > window:
            df['std'] = df['value'].rolling(window=window, center=True).std()
            ax.fill_between(df['step'],
                           df['smoothed'] - df['std'],
                           df['smoothed'] + df['std'],
                           alpha=0.15, color='coral')

        # Add stage boundaries
        if stage_info:
            for i, stage in enumerate(stage_info):
                if i > 0:
                    ax.axvline(x=stage['start'], color='gray', linestyle='--',
                              linewidth=1.5, alpha=0.7)

                mid_x = (stage['start'] + stage['end']) / 2
                ax.text(mid_x, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 900,
                       stage['name'].split(':')[0],
                       ha='center', va='top', fontsize=10, fontweight='bold',
                       color=stage['color'],
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        ax.set_xlabel('Training Steps', fontweight='bold')
        ax.set_ylabel('Mean Episode Length (steps)', fontweight='bold')
        ax.set_title('Curriculum Learning: Episode Length Progression Across All Stages',
                    fontweight='bold', pad=15)

        # Format x-axis
        ax.set_xlim([0, 1000000])
        ax.set_xticks([0, 200000, 500000, 1000000])
        ax.set_xticklabels(['0', '200k', '500k', '1M'])

        # Annotate initial and final
        initial_len = df['value'].iloc[:5].mean()
        final_len = df['value'].iloc[-5:].mean()

        ax.text(50000, initial_len + 20, f'Initial: {initial_len:.0f}',
                fontsize=9, color='darkred', fontweight='bold')
        ax.text(800000, final_len + 20, f'Final: {final_len:.0f}',
                fontsize=9, color='darkred', fontweight='bold')

        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

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
    Generate Figure 7: PPO Training Diagnostics

    Shows 4 subplots:
    - Policy gradient loss
    - Value function loss
    - Entropy
    - Explained variance
    """
    print("\n[Figure 7] Generating PPO training diagnostics...")

    # Define metrics to plot with their display names, colors, and whether to negate
    # Note: entropy_loss in SB3 is -ent_coef * entropy, so we negate it to show actual entropy
    metrics_config = [
        ('train/policy_gradient_loss', 'Policy Gradient Loss', 'green', False),
        ('train/value_loss', 'Value Function Loss', 'orange', False),
        ('train/entropy_loss', 'Entropy', 'purple', True),  # negate=True to show positive entropy
        ('train/explained_variance', 'Explained Variance', 'blue', False)
    ]

    # Check which metrics are available
    available_metrics = []
    for tag, name, color, negate in metrics_config:
        if tag in data:
            available_metrics.append((tag, name, color, negate))
        else:
            # Try alternative names
            alt_tags = {
                'train/policy_gradient_loss': ['train/policy_loss'],
                'train/entropy_loss': ['train/entropy'],
            }
            found = False
            for alt_tag in alt_tags.get(tag, []):
                if alt_tag in data:
                    available_metrics.append((alt_tag, name, color, negate))
                    found = True
                    break
            if not found:
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

        # Negate values if needed (for entropy_loss -> entropy)
        if negate:
            df['value'] = -df['value']

        # Smooth for cleaner visualization
        window = min(10, max(1, len(df) // 20))
        if window > 1 and len(df) > window:
            df['smoothed'] = df['value'].rolling(window=window, center=True).mean()
            ax.plot(df['step'], df['value'], alpha=0.2, color=color, linewidth=0.5)
            ax.plot(df['step'], df['smoothed'], color=color, linewidth=2)
        else:
            ax.plot(df['step'], df['value'], color=color, linewidth=1.5)

        ax.set_xlabel('Training Steps')
        ax.set_ylabel(name)
        ax.set_title(f'{name} over Training')
        ax.grid(True, alpha=0.3)

        # Add trend annotation
        if len(df) > 10:
            initial = df['value'].iloc[:5].mean()
            final = df['value'].iloc[-5:].mean()
            trend = "increasing" if final > initial else "decreasing"
            change = ((final - initial) / abs(initial) * 100) if initial != 0 else 0
            ax.text(0.98, 0.98, f'{trend}\n({change:+.0f}%)',
                   transform=ax.transAxes, ha='right', va='top', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

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
        description='Generate Thesis Figures 5, 6, and 7 from TensorBoard logs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate figures from ALL curriculum stages (recommended)
  python generate_thesis_figures.py \\
      --curriculum ~/training_curriculum_1M \\
      --output ~/Documents/rl-factory-thesis-skeleton/figures/curriculum

  # Generate figures from a single TensorBoard directory
  python generate_thesis_figures.py \\
      --tensorboard ~/training_extended_1M/tensorboard \\
      --output ~/figures/extended
        """
    )
    parser.add_argument('--tensorboard', type=str, default=None,
                        help='Path to single TensorBoard logs directory')
    parser.add_argument('--curriculum', type=str, default=None,
                        help='Path to curriculum training base directory (contains stage_simple, stage_medium, stage_full)')
    parser.add_argument('--output', type=str, default='./figures',
                        help='Output directory for figures (default: ./figures)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("Thesis Figure Generation (Figures 5, 6, 7)")
    print("=" * 60)

    # Determine data source
    if args.curriculum:
        print(f"Curriculum base dir: {args.curriculum}")
        print("Loading ALL 3 curriculum stages...")
        print("=" * 60)

        if not os.path.exists(args.curriculum):
            print(f"\nError: Curriculum directory not found: {args.curriculum}")
            return 1

        data, stage_info = load_all_curriculum_stages(args.curriculum)
        if not data:
            print("\nFailed to load curriculum data")
            return 1

        print(f"\nLoaded {len(stage_info)} stages:")
        for stage in stage_info:
            print(f"  - {stage['name']}: {stage['start']:,} to {stage['end']:,} steps")

    elif args.tensorboard:
        print(f"TensorBoard logs: {args.tensorboard}")
        print("=" * 60)

        if not os.path.exists(args.tensorboard):
            print(f"\nError: TensorBoard directory not found: {args.tensorboard}")
            return 1

        data = load_tensorboard_data(args.tensorboard)
        stage_info = None
        if data is None:
            print("\nFailed to load TensorBoard data")
            return 1
    else:
        print("Error: Must specify either --curriculum or --tensorboard")
        return 1

    print(f"Output directory: {args.output}")
    print("=" * 60)

    # Generate figures
    results = {
        'Figure 5 (learning_curve_reward.png)': generate_figure_5(data, args.output, stage_info),
        'Figure 6 (learning_curve_length.png)': generate_figure_6(data, args.output, stage_info),
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
