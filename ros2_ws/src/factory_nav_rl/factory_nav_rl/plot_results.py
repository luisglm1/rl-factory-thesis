"""
Plotting Script for Training Results and Thesis Figures
Generates publication-quality plots from TensorBoard logs and evaluation results

Usage:
    python plot_results.py --tensorboard /path/to/logs --output ./figures
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing import event_accumulator

# Set publication-quality defaults
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
    'savefig.bbox': 'tight'
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
        print(f"❌ No TensorBoard event file found in {logdir}")
        return None

    print(f"Found event file: {event_file}")

    # Load events
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()

    data = {}

    # Extract scalar data
    tags = ea.Tags()['scalars']
    print(f"Found {len(tags)} scalar metrics")

    for tag in tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        data[tag] = pd.DataFrame({'step': steps, 'value': values})

    return data


def plot_learning_curve(data, output_dir):
    """Plot reward learning curve"""
    print("\n[1/5] Generating learning curve...")

    fig, ax = plt.subplots(figsize=(10, 6))

    if 'rollout/ep_rew_mean' in data:
        df = data['rollout/ep_rew_mean']

        # Smooth curve using rolling average
        window = min(20, len(df) // 10)
        if window > 1:
            df['smoothed'] = df['value'].rolling(window=window, center=True).mean()

        ax.plot(df['step'], df['value'], alpha=0.3, color='steelblue', label='Raw')
        if 'smoothed' in df.columns:
            ax.plot(df['step'], df['smoothed'], linewidth=2, color='darkblue', label='Smoothed')

        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Episode Reward')
        ax.set_title('PPO Training Progress: Episode Reward')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add annotations
        final_reward = df['value'].iloc[-1]
        ax.axhline(y=final_reward, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax.text(df['step'].max() * 0.7, final_reward, f'Final: {final_reward:.1f}',
                verticalalignment='bottom')

        filepath = os.path.join(output_dir, 'learning_curve_reward.png')
        plt.savefig(filepath)
        plt.close()
        print(f"Saved: {filepath}")
    else:
        print("⚠️  Warning: No reward data found")


def plot_episode_length(data, output_dir):
    """Plot episode length over training"""
    print("\n[2/5] Generating episode length plot...")

    fig, ax = plt.subplots(figsize=(10, 6))

    if 'rollout/ep_len_mean' in data:
        df = data['rollout/ep_len_mean']

        # Smooth curve
        window = min(20, len(df) // 10)
        if window > 1:
            df['smoothed'] = df['value'].rolling(window=window, center=True).mean()

        ax.plot(df['step'], df['value'], alpha=0.3, color='coral', label='Raw')
        if 'smoothed' in df.columns:
            ax.plot(df['step'], df['smoothed'], linewidth=2, color='darkred', label='Smoothed')

        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Episode Length (steps)')
        ax.set_title('PPO Training Progress: Episode Length')
        ax.legend()
        ax.grid(True, alpha=0.3)

        filepath = os.path.join(output_dir, 'learning_curve_length.png')
        plt.savefig(filepath)
        plt.close()
        print(f"Saved: {filepath}")
    else:
        print("⚠️  Warning: No episode length data found")


def plot_training_metrics(data, output_dir):
    """Plot additional training metrics (loss, entropy, etc.)"""
    print("\n[3/5] Generating training metrics plot...")

    metrics_to_plot = [
        ('train/loss', 'Total Loss', 'purple'),
        ('train/policy_gradient_loss', 'Policy Gradient Loss', 'green'),
        ('train/value_loss', 'Value Loss', 'orange'),
        ('train/entropy_loss', 'Entropy Loss', 'brown')
    ]

    available_metrics = [(tag, name, color) for tag, name, color in metrics_to_plot if tag in data]

    if not available_metrics:
        print("⚠️  Warning: No training metrics found")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (tag, name, color) in enumerate(available_metrics[:4]):
        ax = axes[idx]
        df = data[tag]

        ax.plot(df['step'], df['value'], color=color, alpha=0.7, linewidth=1.5)
        ax.set_xlabel('Training Steps')
        ax.set_ylabel(name)
        ax.set_title(f'{name} over Training')
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(available_metrics), 4):
        axes[idx].axis('off')

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'training_metrics.png')
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath}")


def plot_evaluation_results(eval_summary_path, output_dir):
    """Plot evaluation results"""
    print("\n[4/5] Generating evaluation results plot...")

    if not os.path.exists(eval_summary_path):
        print(f"⚠️  Warning: Evaluation summary not found: {eval_summary_path}")
        return

    df = pd.read_csv(eval_summary_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Outcome distribution
    ax = axes[0]
    outcomes = ['Success', 'Collision', 'Timeout', 'Out of Bounds']
    rates = [
        df['success_rate'].values[0],
        df['collision_rate'].values[0],
        df['timeout_rate'].values[0],
        df['out_of_bounds_rate'].values[0]
    ]
    colors = ['green', 'red', 'orange', 'purple']

    ax.bar(outcomes, rates, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Rate (%)')
    ax.set_title('Episode Outcomes (100 Episodes)')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 100])

    # Add value labels on bars
    for i, (outcome, rate) in enumerate(zip(outcomes, rates)):
        ax.text(i, rate + 2, f'{rate:.1f}%', ha='center', fontweight='bold')

    # Reward distribution (placeholder - would need episode data)
    ax = axes[1]
    ax.text(0.5, 0.5, 'Reward distribution\n(requires episode data)',
            ha='center', va='center', fontsize=12, color='gray')
    ax.set_title('Reward Distribution')
    ax.axis('off')

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'evaluation_results.png')
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath}")


def create_summary_figure(data, eval_data, output_dir):
    """Create a comprehensive summary figure for the thesis"""
    print("\n[5/5] Generating comprehensive summary figure...")

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Learning curve (top row, full width)
    ax1 = fig.add_subplot(gs[0, :])
    if 'rollout/ep_rew_mean' in data:
        df = data['rollout/ep_rew_mean']
        window = min(20, len(df) // 10)
        if window > 1:
            df['smoothed'] = df['value'].rolling(window=window, center=True).mean()
            ax1.plot(df['step'], df['smoothed'], linewidth=2, color='darkblue', label='Episode Reward')
        else:
            ax1.plot(df['step'], df['value'], linewidth=2, color='darkblue', label='Episode Reward')
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Mean Episode Reward')
        ax1.set_title('A) Training Progress', fontweight='bold', loc='left')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # Episode length (middle left)
    ax2 = fig.add_subplot(gs[1, 0])
    if 'rollout/ep_len_mean' in data:
        df = data['rollout/ep_len_mean']
        ax2.plot(df['step'], df['value'], color='coral', alpha=0.6)
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Episode Length')
        ax2.set_title('B) Episode Duration', fontweight='bold', loc='left')
        ax2.grid(True, alpha=0.3)

    # Loss (middle center)
    ax3 = fig.add_subplot(gs[1, 1])
    if 'train/loss' in data:
        df = data['train/loss']
        ax3.plot(df['step'], df['value'], color='purple', alpha=0.6)
        ax3.set_xlabel('Steps')
        ax3.set_ylabel('Loss')
        ax3.set_title('C) Training Loss', fontweight='bold', loc='left')
        ax3.grid(True, alpha=0.3)

    # Entropy (middle right)
    ax4 = fig.add_subplot(gs[1, 2])
    if 'train/entropy_loss' in data:
        df = data['train/entropy_loss']
        ax4.plot(df['step'], df['value'], color='brown', alpha=0.6)
        ax4.set_xlabel('Steps')
        ax4.set_ylabel('Entropy')
        ax4.set_title('D) Exploration', fontweight='bold', loc='left')
        ax4.grid(True, alpha=0.3)

    # Performance metrics table (bottom row)
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')

    if eval_data and os.path.exists(eval_data):
        summary = pd.read_csv(eval_data)
        table_data = [
            ['Success Rate', f"{summary['success_rate'].values[0]:.1f}%"],
            ['Collision Rate', f"{summary['collision_rate'].values[0]:.1f}%"],
            ['Mean Reward', f"{summary['mean_reward'].values[0]:.1f} ± {summary['std_reward'].values[0]:.1f}"],
            ['Mean Episode Length', f"{summary['mean_length'].values[0]:.0f} ± {summary['std_length'].values[0]:.0f}"]
        ]

        table = ax5.table(cellText=table_data, colLabels=['Metric', 'Value'],
                         cellLoc='left', loc='center',
                         bbox=[0.25, 0.3, 0.5, 0.6])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)
        ax5.set_title('E) Final Performance (100 Episodes)', fontweight='bold', loc='left', pad=20)

    fig.suptitle('PPO Training Results: Factory Navigation Task', fontsize=16, fontweight='bold', y=0.98)

    filepath = os.path.join(output_dir, 'summary_figure.png')
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath}")


def main():
    parser = argparse.ArgumentParser(description='Generate plots from training results')
    parser.add_argument('--tensorboard', type=str, required=True,
                        help='Path to TensorBoard logs directory')
    parser.add_argument('--eval', type=str, default=None,
                        help='Path to evaluation summary CSV (optional)')
    parser.add_argument('--output', type=str, default='./figures',
                        help='Output directory for figures (default: ./figures)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("Training Results Plotting")
    print("=" * 60)
    print(f"TensorBoard logs: {args.tensorboard}")
    print(f"Output directory: {args.output}")
    print("=" * 60)

    # Load data
    data = load_tensorboard_data(args.tensorboard)
    if data is None:
        print("❌ Failed to load TensorBoard data")
        return

    # Generate plots
    plot_learning_curve(data, args.output)
    plot_episode_length(data, args.output)
    plot_training_metrics(data, args.output)

    if args.eval:
        plot_evaluation_results(args.eval, args.output)

    # Create summary figure
    create_summary_figure(data, args.eval, args.output)

    print("\n" + "=" * 60)
    print("All figures generated successfully!")
    print(f"Figures saved to: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
