"""Visualization script for VGG masked loss experiments.

Generates plots comparing baseline vs masked loss training:
- Generalization gap over epochs
- Test accuracy over epochs
- Loss curves (L_full, L_masked_avg, L_total)

Aggregates results across multiple seeds (mean ± std).
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set publication-quality plotting defaults
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.size'] = 10
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['axes.labelsize'] = 10
mpl.rcParams['axes.titlesize'] = 11
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9
mpl.rcParams['legend.fontsize'] = 9


def load_results(
    results_dir: Path,
    arch: str,
    mode: str,
    seeds: List[int]
) -> List[Dict]:
    """Load results JSON files for given architecture, mode, and seeds.

    Args:
        results_dir: Directory containing results
        arch: Architecture name (e.g., 'vgg11')
        mode: Training mode ('baseline' or 'masked')
        seeds: List of seed values

    Returns:
        List of result dictionaries (one per seed)
    """
    results = []

    for seed in seeds:
        results_file = results_dir / f"{arch}_{mode}_seed{seed}.json"

        if not results_file.exists():
            print(f"Warning: {results_file} not found, skipping...")
            continue

        with open(results_file, 'r') as f:
            data = json.load(f)
            results.append(data)

    return results


def aggregate_histories(results: List[Dict], key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate a metric across multiple seeds.

    Args:
        results: List of result dictionaries
        key: History key to aggregate (e.g., 'gen_gap', 'test_acc')

    Returns:
        (epochs, mean_values, std_values) as numpy arrays
    """
    if not results:
        return np.array([]), np.array([]), np.array([])

    # Extract histories for this key
    histories = [np.array(r['history'][key]) for r in results]

    # All runs should have same number of epochs
    epochs = np.array(results[0]['history']['epoch'])

    # Stack and compute statistics
    stacked = np.stack(histories, axis=0)  # (n_seeds, n_epochs)
    mean_values = stacked.mean(axis=0)
    std_values = stacked.std(axis=0)

    return epochs, mean_values, std_values


def plot_generalization_gap(
    results_dir: Path,
    architectures: List[str],
    modes: List[str],
    seeds: List[int],
    output_dir: Path
):
    """Plot generalization gap over epochs for all architectures and modes.

    Creates one plot with all architectures, comparing baseline vs masked.

    Args:
        results_dir: Directory containing results
        architectures: List of architectures (e.g., ['vgg11', 'vgg13', 'vgg16', 'vgg19'])
        modes: List of modes (e.g., ['baseline', 'masked'])
        seeds: List of seeds to aggregate over
        output_dir: Directory to save plots
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {
        'vgg11': '#1f77b4',
        'vgg13': '#ff7f0e',
        'vgg16': '#2ca02c',
        'vgg19': '#d62728',
    }

    linestyles = {
        'baseline': '-',
        'masked': '--',
    }

    for arch in architectures:
        for mode in modes:
            # Load results
            results = load_results(results_dir, arch, mode, seeds)

            if not results:
                print(f"No results found for {arch} {mode}, skipping...")
                continue

            # Aggregate
            epochs, mean_gap, std_gap = aggregate_histories(results, 'gen_gap')

            # Plot
            label = f"{arch.upper()} ({mode})"
            color = colors[arch]
            linestyle = linestyles[mode]

            ax.plot(epochs, mean_gap, label=label, color=color, linestyle=linestyle, linewidth=1.5)
            ax.fill_between(epochs, mean_gap - std_gap, mean_gap + std_gap,
                           color=color, alpha=0.15)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Generalization Gap (%)')
    ax.set_title('Generalization Gap: Baseline vs Masked Loss Training')
    ax.legend(loc='best', ncol=2)
    ax.grid(True, alpha=0.3)

    # Save
    output_path = output_dir / 'generalization_gap.png'
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved: {output_path}")


def plot_test_accuracy(
    results_dir: Path,
    architectures: List[str],
    modes: List[str],
    seeds: List[int],
    output_dir: Path
):
    """Plot test accuracy over epochs for all architectures and modes.

    Args:
        results_dir: Directory containing results
        architectures: List of architectures
        modes: List of modes
        seeds: List of seeds
        output_dir: Directory to save plots
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {
        'vgg11': '#1f77b4',
        'vgg13': '#ff7f0e',
        'vgg16': '#2ca02c',
        'vgg19': '#d62728',
    }

    linestyles = {
        'baseline': '-',
        'masked': '--',
    }

    for arch in architectures:
        for mode in modes:
            results = load_results(results_dir, arch, mode, seeds)

            if not results:
                continue

            epochs, mean_acc, std_acc = aggregate_histories(results, 'test_acc')

            label = f"{arch.upper()} ({mode})"
            color = colors[arch]
            linestyle = linestyles[mode]

            ax.plot(epochs, mean_acc, label=label, color=color, linestyle=linestyle, linewidth=1.5)
            ax.fill_between(epochs, mean_acc - std_acc, mean_acc + std_acc,
                           color=color, alpha=0.15)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Test Accuracy: Baseline vs Masked Loss Training')
    ax.legend(loc='best', ncol=2)
    ax.grid(True, alpha=0.3)

    output_path = output_dir / 'test_accuracy.png'
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved: {output_path}")


def plot_loss_curves(
    results_dir: Path,
    architectures: List[str],
    seeds: List[int],
    output_dir: Path
):
    """Plot loss curves (L_full, L_masked_avg, L_total) for masked training only.

    Creates separate plots for each architecture.

    Args:
        results_dir: Directory containing results
        architectures: List of architectures
        seeds: List of seeds
        output_dir: Directory to save plots
    """
    for arch in architectures:
        # Only plot for masked mode
        results = load_results(results_dir, arch, 'masked', seeds)

        if not results:
            print(f"No masked results found for {arch}, skipping loss curves...")
            continue

        # Aggregate loss components
        epochs, mean_full, std_full = aggregate_histories(results, 'train_loss_full')
        _, mean_masked, std_masked = aggregate_histories(results, 'train_loss_masked')
        _, mean_total, std_total = aggregate_histories(results, 'train_loss_total')

        # Plot
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(epochs, mean_full, label='L_full', color='blue', linewidth=1.5)
        ax.fill_between(epochs, mean_full - std_full, mean_full + std_full,
                       color='blue', alpha=0.15)

        ax.plot(epochs, mean_masked, label='L_masked_avg', color='red', linewidth=1.5)
        ax.fill_between(epochs, mean_masked - std_masked, mean_masked + std_masked,
                       color='red', alpha=0.15)

        ax.plot(epochs, mean_total, label='L_total (optimized)', color='green', linewidth=1.5)
        ax.fill_between(epochs, mean_total - std_total, mean_total + std_total,
                       color='green', alpha=0.15)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'Loss Components: {arch.upper()} (Masked Training)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        output_path = output_dir / f'loss_curves_{arch}.png'
        fig.tight_layout()
        fig.savefig(output_path, bbox_inches='tight')
        plt.close(fig)

        print(f"Saved: {output_path}")


def plot_lambda_schedule(
    results_dir: Path,
    architectures: List[str],
    seeds: List[int],
    output_dir: Path
):
    """Plot lambda warmup schedule for masked training.

    Args:
        results_dir: Directory containing results
        architectures: List of architectures
        seeds: List of seeds
        output_dir: Directory to save plots
    """
    # Just use first architecture and seed to show schedule
    arch = architectures[0]
    results = load_results(results_dir, arch, 'masked', seeds[:1])

    if not results:
        print("No masked results found, skipping lambda schedule plot...")
        return

    epochs = np.array(results[0]['history']['epoch'])
    lambda_values = np.array(results[0]['history']['lambda'])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, lambda_values, color='purple', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('λ (Lambda)')
    ax.set_title('Lambda Warmup Schedule')
    ax.grid(True, alpha=0.3)

    output_path = output_dir / 'lambda_schedule.png'
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved: {output_path}")


def print_summary_table(
    results_dir: Path,
    architectures: List[str],
    modes: List[str],
    seeds: List[int]
):
    """Print a summary table of final results.

    Args:
        results_dir: Directory containing results
        architectures: List of architectures
        modes: List of modes
        seeds: List of seeds
    """
    print("\n" + "="*80)
    print("SUMMARY OF FINAL RESULTS (mean ± std)")
    print("="*80)
    print(f"{'Architecture':<12} {'Mode':<10} {'Test Acc (%)':<15} {'Gen Gap (%)':<15}")
    print("-"*80)

    for arch in architectures:
        for mode in modes:
            results = load_results(results_dir, arch, mode, seeds)

            if not results:
                continue

            # Extract final values
            final_test_accs = [r['final_test_acc'] for r in results]
            final_gen_gaps = [r['final_gen_gap'] for r in results]

            mean_test_acc = np.mean(final_test_accs)
            std_test_acc = np.std(final_test_accs)

            mean_gen_gap = np.mean(final_gen_gaps)
            std_gen_gap = np.std(final_gen_gaps)

            print(f"{arch.upper():<12} {mode:<10} "
                  f"{mean_test_acc:.2f} ± {std_test_acc:.2f}    "
                  f"{mean_gen_gap:.2f} ± {std_gen_gap:.2f}")

    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Visualize VGG masked loss experiment results')

    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory containing result JSON files')
    parser.add_argument('--plots_dir', type=str, default='./plots',
                        help='Directory to save plots')
    parser.add_argument('--architectures', type=str, nargs='+',
                        default=['vgg11', 'vgg13', 'vgg16', 'vgg19'],
                        help='Architectures to plot')
    parser.add_argument('--modes', type=str, nargs='+',
                        default=['baseline', 'masked'],
                        help='Training modes to plot')
    parser.add_argument('--seeds', type=int, nargs='+',
                        default=[0, 1, 2],
                        help='Seeds to aggregate over')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("Generating visualizations...")
    print(f"Results directory: {results_dir}")
    print(f"Plots directory: {plots_dir}")
    print(f"Architectures: {args.architectures}")
    print(f"Modes: {args.modes}")
    print(f"Seeds: {args.seeds}")
    print()

    # Generate plots
    plot_generalization_gap(results_dir, args.architectures, args.modes, args.seeds, plots_dir)
    plot_test_accuracy(results_dir, args.architectures, args.modes, args.seeds, plots_dir)
    plot_loss_curves(results_dir, args.architectures, args.seeds, plots_dir)
    plot_lambda_schedule(results_dir, args.architectures, args.seeds, plots_dir)

    # Print summary
    print_summary_table(results_dir, args.architectures, args.modes, args.seeds)

    print("Done! All plots saved to:", plots_dir)


if __name__ == '__main__':
    main()
