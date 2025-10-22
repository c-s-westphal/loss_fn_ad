"""Training script for VGG models with masked loss objective.

Trains VGG models (11/13/16/19) on CIFAR-10 with either:
- Baseline: Standard cross-entropy loss
- Masked: Cross-entropy loss with neuron masking objective

The masked loss objective encourages robustness by maximizing loss when
random subsets of neurons are masked (zeroed out) in the first convolutional layer.

Loss formulation:
    L_total = L_full - λ(t) * min(L_masked_avg, 0.5 * L_full)

where λ(t) warms up linearly from 0 to 1.0 over the first 20% of training.
"""

import os
import argparse
import json
from pathlib import Path
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

from models.vgg_cifar import VGG11, VGG13, VGG16, VGG19


def get_data_loaders(
    batch_size: int = 128,
    num_workers: int = 4,
    data_root: str = './data'
) -> Tuple[DataLoader, DataLoader]:
    """Create CIFAR-10 train and test data loaders.

    Standard augmentation: RandomCrop + RandomHorizontalFlip
    Normalization using CIFAR-10 statistics.
    """
    # Normalization statistics for CIFAR-10
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )

    # Training transform with augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    # Test transform (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=train_transform
    )

    test_dataset = datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, test_loader


def generate_random_neuron_masks(n_neurons: int, n_masks: int) -> List[np.ndarray]:
    """Generate random masks for neuron selection in the first convolutional layer.

    Each mask randomly selects between 1 and (n_neurons - 2) neurons to keep.
    Returns masks where True = keep neuron, False = zero out neuron.

    Args:
        n_neurons: Number of neurons (feature maps) in the layer
        n_masks: Number of different masks to generate

    Returns:
        List of boolean arrays of shape (n_neurons,)
    """
    masks = []
    max_subset_size = max(1, n_neurons - 2)

    for _ in range(n_masks):
        # Random subset size between 1 and max_subset_size
        subset_size = np.random.randint(1, max_subset_size + 1)

        # Create mask and randomly select neurons to KEEP
        mask = np.zeros(n_neurons, dtype=bool)
        selected_indices = np.random.choice(n_neurons, subset_size, replace=False)
        mask[selected_indices] = True
        masks.append(mask)

    return masks


class NeuronMaskingHook:
    """Forward hook to mask (zero out) specific neurons/feature maps in a layer."""

    def __init__(self, mask: np.ndarray):
        """
        Args:
            mask: Boolean array where True = keep neuron, False = zero out
                  Shape: (n_neurons,) or (n_channels,)
        """
        self.mask = torch.from_numpy(mask).bool()

    def __call__(self, module, input, output):
        """Apply neuron masking during forward pass.

        Args:
            output: Tensor of shape (batch, n_neurons) for FC layers
                    or (batch, channels, height, width) for Conv layers

        Returns:
            Masked output
        """
        # Clone to avoid in-place modification issues
        masked_output = output.clone()

        # Handle different tensor dimensions
        if output.dim() == 2:
            # FC layer: (batch, n_neurons)
            # Broadcast mask across batch dimension: (1, n_neurons)
            mask_broadcast = self.mask.unsqueeze(0).to(output.device)
        elif output.dim() == 4:
            # Conv layer: (batch, channels, height, width)
            # Broadcast mask across batch and spatial dimensions: (1, channels, 1, 1)
            mask_broadcast = self.mask.view(1, -1, 1, 1).to(output.device)
        else:
            raise ValueError(f"Unexpected output dimension: {output.dim()}")

        masked_output = masked_output * mask_broadcast
        return masked_output


def compute_loss_with_masking(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    masking_layer: nn.Module,
    n_masks: int = 3
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute full loss and average masked loss.

    Args:
        model: The neural network model
        inputs: Input batch (batch, 3, 32, 32)
        targets: Target labels (batch,)
        criterion: Loss function
        masking_layer: Layer where masking should be applied
        n_masks: Number of random masks to average over

    Returns:
        (loss_full, loss_masked_avg, loss_total)
    """
    # 1. Forward pass without masking (full model)
    outputs_full = model(inputs)
    loss_full = criterion(outputs_full, targets)

    # 2. Get number of neurons to mask
    # Do a forward pass to get the shape
    sample_output = None
    def capture_hook(module, input, output):
        nonlocal sample_output
        sample_output = output

    handle = masking_layer.register_forward_hook(capture_hook)
    with torch.no_grad():
        _ = model(inputs[:1])
    handle.remove()

    n_neurons = sample_output.shape[1]

    # 3. Generate random masks and compute masked losses
    masks = generate_random_neuron_masks(n_neurons, n_masks)
    masked_losses = []

    for mask in masks:
        # Attach masking hook
        hook = NeuronMaskingHook(mask)
        hook_handle = masking_layer.register_forward_hook(hook)

        # Forward pass with masking
        outputs_masked = model(inputs)
        loss_masked = criterion(outputs_masked, targets)
        masked_losses.append(loss_masked)

        # Remove hook
        hook_handle.remove()

    # 4. Average masked losses
    loss_masked_avg = torch.stack(masked_losses).mean()

    return loss_full, loss_masked_avg


def compute_masked_loss_objective(
    loss_full: torch.Tensor,
    loss_masked_avg: torch.Tensor,
    lambda_current: float
) -> torch.Tensor:
    """Compute the masked loss training objective.

    L_total = L_full - λ * min(L_masked_avg, 0.5 * L_full)

    The clamping ensures the masked term never dominates.

    Args:
        loss_full: Loss from full model
        loss_masked_avg: Average loss from masked models
        lambda_current: Current lambda value (warmup schedule)

    Returns:
        Total loss to optimize
    """
    # Clamp masked term to at most 50% of full loss
    masked_term = lambda_current * torch.min(
        loss_masked_avg,
        0.5 * loss_full
    )

    loss_total = loss_full - masked_term

    return loss_total


def get_lambda_warmup(epoch: int, total_epochs: int, lambda_max: float = 1.0) -> float:
    """Compute lambda value with linear warmup.

    Lambda warms up from 0 to lambda_max over the first 20% of training,
    then stays constant at lambda_max.

    Args:
        epoch: Current epoch (1-indexed)
        total_epochs: Total number of training epochs
        lambda_max: Maximum lambda value (default 1.0)

    Returns:
        Current lambda value
    """
    warmup_epochs = int(0.2 * total_epochs)

    if epoch <= warmup_epochs:
        return lambda_max * (epoch / warmup_epochs)
    else:
        return lambda_max


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    use_masked_loss: bool = False,
    total_epochs: int = 200,
    n_masks: int = 3
) -> Tuple[float, float, float, float]:
    """Train for one epoch.

    Args:
        model: Neural network model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use
        epoch: Current epoch number (1-indexed)
        use_masked_loss: Whether to use masked loss objective
        total_epochs: Total number of epochs (for lambda warmup)
        n_masks: Number of masks to use if using masked loss

    Returns:
        (avg_loss_full, avg_loss_masked, avg_loss_total, accuracy)
    """
    model.train()

    total_loss_full = 0.0
    total_loss_masked = 0.0
    total_loss_total = 0.0
    correct = 0
    total = 0

    # Get masking layer if needed
    masking_layer = model.get_masking_layer() if use_masked_loss else None

    # Get current lambda value
    lambda_current = get_lambda_warmup(epoch, total_epochs) if use_masked_loss else 0.0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        if use_masked_loss:
            # Compute masked loss objective
            loss_full, loss_masked_avg = compute_loss_with_masking(
                model, inputs, targets, criterion, masking_layer, n_masks
            )

            loss_total = compute_masked_loss_objective(
                loss_full, loss_masked_avg, lambda_current
            )

            # Backprop through total loss
            loss_total.backward()

            # Track all loss components
            total_loss_full += loss_full.item()
            total_loss_masked += loss_masked_avg.item()
            total_loss_total += loss_total.item()
        else:
            # Standard training
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            total_loss_full += loss.item()
            total_loss_masked += 0.0
            total_loss_total += loss.item()

        optimizer.step()

        # Calculate accuracy (using full model, no masking)
        with torch.no_grad():
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Compute averages
    num_batches = len(train_loader)
    avg_loss_full = total_loss_full / num_batches
    avg_loss_masked = total_loss_masked / num_batches
    avg_loss_total = total_loss_total / num_batches
    accuracy = 100.0 * correct / total

    return avg_loss_full, avg_loss_masked, avg_loss_total, accuracy


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device
) -> float:
    """Evaluate model accuracy on test set.

    Args:
        model: Neural network model
        test_loader: Test data loader
        device: Device to use

    Returns:
        Test accuracy (percentage)
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100.0 * correct / total
    return accuracy


def main():
    parser = argparse.ArgumentParser(
        description='Train VGG on CIFAR-10 with optional masked loss objective'
    )

    # Model arguments
    parser.add_argument('--arch', type=str, required=True,
                        choices=['vgg11', 'vgg13', 'vgg16', 'vgg19'],
                        help='Model architecture')
    parser.add_argument('--seed', type=int, required=True,
                        help='Random seed')

    # Training mode
    parser.add_argument('--mode', type=str, required=True,
                        choices=['baseline', 'masked'],
                        help='Training mode: baseline (standard) or masked (with masking objective)')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay')

    # Masked loss hyperparameters
    parser.add_argument('--n_masks', type=int, default=3,
                        help='Number of masks per batch (for masked mode)')
    parser.add_argument('--lambda_max', type=float, default=1.0,
                        help='Maximum lambda value after warmup')

    # System arguments
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory for CIFAR-10 data')

    # Output arguments
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory to save results')

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print("="*70)
    print(f"Training Configuration")
    print("="*70)
    print(f"Architecture: {args.arch.upper()}")
    print(f"Mode: {args.mode.upper()}")
    print(f"Seed: {args.seed}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    if args.mode == 'masked':
        print(f"Number of masks: {args.n_masks}")
        print(f"Lambda max: {args.lambda_max}")
    print("="*70)

    # Create model
    model_map = {
        'vgg11': VGG11,
        'vgg13': VGG13,
        'vgg16': VGG16,
        'vgg19': VGG19
    }
    model = model_map[args.arch](num_classes=10, use_batchnorm=True)
    model = model.to(device)

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("="*70)

    # Create data loaders
    train_loader, test_loader = get_data_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_root=args.data_dir
    )

    # Setup optimizer (SGD with momentum, standard for CIFAR-10)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    # Learning rate scheduler: MultiStep decay at 50% and 75% of training
    milestones = [int(0.5 * args.epochs), int(0.75 * args.epochs)]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Training history
    history = {
        'epoch': [],
        'train_loss_full': [],
        'train_loss_masked': [],
        'train_loss_total': [],
        'train_acc': [],
        'test_acc': [],
        'gen_gap': [],
        'lambda': [],
    }

    use_masked_loss = (args.mode == 'masked')

    # Training loop
    print("\nStarting training...")
    print("="*70)

    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss_full, train_loss_masked, train_loss_total, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            use_masked_loss=use_masked_loss,
            total_epochs=args.epochs,
            n_masks=args.n_masks
        )

        # Evaluate
        test_acc = evaluate(model, test_loader, device)
        gen_gap = train_acc - test_acc

        # Get current lambda
        lambda_current = get_lambda_warmup(epoch, args.epochs) if use_masked_loss else 0.0

        # Step scheduler
        scheduler.step()

        # Record history
        history['epoch'].append(epoch)
        history['train_loss_full'].append(train_loss_full)
        history['train_loss_masked'].append(train_loss_masked)
        history['train_loss_total'].append(train_loss_total)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['gen_gap'].append(gen_gap)
        history['lambda'].append(lambda_current)

        # Print progress every 10 epochs
        if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
            if use_masked_loss:
                print(f"Epoch {epoch:3d}/{args.epochs}: "
                      f"L_full={train_loss_full:.4f}, L_masked={train_loss_masked:.4f}, "
                      f"L_total={train_loss_total:.4f}, λ={lambda_current:.3f} | "
                      f"Train={train_acc:.2f}%, Test={test_acc:.2f}%, Gap={gen_gap:.2f}%")
            else:
                print(f"Epoch {epoch:3d}/{args.epochs}: "
                      f"Loss={train_loss_total:.4f} | "
                      f"Train={train_acc:.2f}%, Test={test_acc:.2f}%, Gap={gen_gap:.2f}%")

    print("="*70)
    print("Training completed!")
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    print(f"Final Generalization Gap: {gen_gap:.2f}%")
    print("="*70)

    # Save checkpoint
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_name = f"{args.arch}_{args.mode}_seed{args.seed}.pt"
    checkpoint_path = checkpoint_dir / checkpoint_name

    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'arch': args.arch,
        'mode': args.mode,
        'seed': args.seed,
        'final_train_acc': train_acc,
        'final_test_acc': test_acc,
        'final_gen_gap': gen_gap,
    }, checkpoint_path)

    print(f"\nCheckpoint saved: {checkpoint_path}")

    # Save results
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    results_name = f"{args.arch}_{args.mode}_seed{args.seed}.json"
    results_path = results_dir / results_name

    results = {
        'arch': args.arch,
        'mode': args.mode,
        'seed': args.seed,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'n_masks': args.n_masks if use_masked_loss else 0,
        'lambda_max': args.lambda_max if use_masked_loss else 0.0,
        'final_train_acc': float(train_acc),
        'final_test_acc': float(test_acc),
        'final_gen_gap': float(gen_gap),
        'history': history,
    }

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {results_path}")
    print("\nDone!")


if __name__ == '__main__':
    main()
