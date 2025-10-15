"""Script to run all VGG masked loss experiments.

Launches training for:
- 4 architectures: VGG11, VGG13, VGG16, VGG19
- 2 modes: baseline, masked
- 3 seeds: 0, 1, 2
Total: 24 experiments

Can run sequentially or generate parallel launch commands.
"""

import argparse
import subprocess
from pathlib import Path
from typing import List


def generate_command(
    arch: str,
    mode: str,
    seed: int,
    epochs: int = 200,
    batch_size: int = 128,
    lr: float = 0.1,
    n_masks: int = 3,
    device: str = 'cuda'
) -> str:
    """Generate the training command for one experiment.

    Args:
        arch: Architecture name (vgg11/13/16/19)
        mode: Training mode (baseline/masked)
        seed: Random seed
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        n_masks: Number of masks (for masked mode)
        device: Device to use

    Returns:
        Command string
    """
    cmd = (
        f"python train.py "
        f"--arch {arch} "
        f"--mode {mode} "
        f"--seed {seed} "
        f"--epochs {epochs} "
        f"--batch_size {batch_size} "
        f"--lr {lr} "
        f"--n_masks {n_masks} "
        f"--device {device}"
    )
    return cmd


def run_experiments_sequential(
    architectures: List[str],
    modes: List[str],
    seeds: List[int],
    epochs: int,
    batch_size: int,
    lr: float,
    n_masks: int,
    device: str
):
    """Run all experiments sequentially.

    Args:
        architectures: List of architectures
        modes: List of modes
        seeds: List of seeds
        epochs: Number of epochs
        batch_size: Batch size
        lr: Learning rate
        n_masks: Number of masks
        device: Device to use
    """
    total = len(architectures) * len(modes) * len(seeds)
    current = 0

    print("="*80)
    print(f"Running {total} experiments sequentially")
    print("="*80)

    for arch in architectures:
        for mode in modes:
            for seed in seeds:
                current += 1

                print(f"\n[{current}/{total}] Running: {arch} {mode} seed={seed}")
                print("-"*80)

                cmd = generate_command(arch, mode, seed, epochs, batch_size, lr, n_masks, device)
                print(f"Command: {cmd}\n")

                # Run the command
                result = subprocess.run(cmd, shell=True)

                if result.returncode != 0:
                    print(f"ERROR: Experiment failed with return code {result.returncode}")
                    print("Continuing with next experiment...")

                print("-"*80)

    print("\n" + "="*80)
    print("All experiments completed!")
    print("="*80)


def generate_parallel_commands(
    architectures: List[str],
    modes: List[str],
    seeds: List[int],
    epochs: int,
    batch_size: int,
    lr: float,
    n_masks: int,
    device: str,
    output_file: str = 'run_all.sh'
):
    """Generate a shell script with all commands for parallel execution.

    Users can modify this script to run experiments in parallel on a cluster.

    Args:
        architectures: List of architectures
        modes: List of modes
        seeds: List of seeds
        epochs: Number of epochs
        batch_size: Batch size
        lr: Learning rate
        n_masks: Number of masks
        device: Device to use
        output_file: Output shell script filename
    """
    commands = []

    for arch in architectures:
        for mode in modes:
            for seed in seeds:
                cmd = generate_command(arch, mode, seed, epochs, batch_size, lr, n_masks, device)
                commands.append(cmd)

    # Write shell script
    script_path = Path(output_file)

    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Auto-generated script to run all VGG masked loss experiments\n")
        f.write(f"# Total experiments: {len(commands)}\n\n")

        for i, cmd in enumerate(commands, 1):
            f.write(f"# Experiment {i}/{len(commands)}\n")
            f.write(f"{cmd}\n\n")

    # Make executable
    script_path.chmod(0o755)

    print(f"Generated shell script: {script_path}")
    print(f"Total commands: {len(commands)}")
    print(f"\nTo run all experiments sequentially:")
    print(f"  ./{output_file}")
    print(f"\nTo run in parallel (e.g., GNU parallel):")
    print(f"  cat {output_file} | grep '^python' | parallel -j 4")


def main():
    parser = argparse.ArgumentParser(description='Run VGG masked loss experiments')

    # Experiment configuration
    parser.add_argument('--architectures', type=str, nargs='+',
                        default=['vgg11', 'vgg13', 'vgg16', 'vgg19'],
                        help='Architectures to train')
    parser.add_argument('--modes', type=str, nargs='+',
                        default=['baseline', 'masked'],
                        help='Training modes')
    parser.add_argument('--seeds', type=int, nargs='+',
                        default=[0, 1, 2],
                        help='Random seeds')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate')
    parser.add_argument('--n_masks', type=int, default=3,
                        help='Number of masks per batch')

    # Execution mode
    parser.add_argument('--mode', type=str, default='sequential',
                        choices=['sequential', 'generate'],
                        help='Execution mode: sequential (run now) or generate (create shell script)')
    parser.add_argument('--output_script', type=str, default='run_all.sh',
                        help='Output shell script name (for generate mode)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    args = parser.parse_args()

    print("Experiment Configuration:")
    print(f"  Architectures: {args.architectures}")
    print(f"  Modes: {args.modes}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Number of masks: {args.n_masks}")
    print(f"  Device: {args.device}")
    print()

    if args.mode == 'sequential':
        run_experiments_sequential(
            args.architectures,
            args.modes,
            args.seeds,
            args.epochs,
            args.batch_size,
            args.lr,
            args.n_masks,
            args.device
        )
    elif args.mode == 'generate':
        generate_parallel_commands(
            args.architectures,
            args.modes,
            args.seeds,
            args.epochs,
            args.batch_size,
            args.lr,
            args.n_masks,
            args.device,
            args.output_script
        )


if __name__ == '__main__':
    main()
