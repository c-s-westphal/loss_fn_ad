# VGG Masked Loss Training

Training VGG models on CIFAR-10 with a novel masked loss objective that encourages neuron-level robustness.

## Overview

This repository implements a training approach that maximizes the loss when random subsets of neurons are masked (zeroed out) in the final fully-connected layer, while minimizing the standard cross-entropy loss. The goal is to make the model more robust and less reliant on specific neurons.

### Masked Loss Objective

The total loss is:

```
L_total = L_full - λ(t) * min(L_masked_avg, 0.5 * L_full)
```

Where:
- `L_full`: Standard cross-entropy loss on the full model
- `L_masked_avg`: Average cross-entropy loss across multiple random neuron masks
- `λ(t)`: Lambda parameter with linear warmup from 0 to 1.0 over first 20% of training
- Clamping ensures the masked term never exceeds 50% of `L_full`

### Masking Strategy

- **Location**: Neurons in the first FC layer of the classifier (after ReLU activation)
- **Masks per batch**: 3 random masks
- **Mask density**: Each mask randomly keeps between 1 and (n_neurons - 2) neurons
- **Regeneration**: New masks generated every iteration

## Repository Structure

```
loss_fn_ad/
├── models/              # VGG model implementations
│   ├── __init__.py
│   └── vgg_cifar.py    # VGG11/13/16/19 for CIFAR-10
├── data/               # CIFAR-10 dataset (auto-downloaded)
├── results/            # JSON files with training history
├── checkpoints/        # Model checkpoints
├── plots/              # Generated plots
├── train.py            # Training script
├── visualize.py        # Visualization script
├── run_experiments.py  # Experiment runner
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Installation

```bash
# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training a Single Model

Train a baseline VGG11:
```bash
python train.py --arch vgg11 --mode baseline --seed 0
```

Train VGG11 with masked loss:
```bash
python train.py --arch vgg11 --mode masked --seed 0
```

### Running All Experiments

The full experiment suite trains:
- 4 architectures: VGG11, VGG13, VGG16, VGG19
- 2 modes: baseline, masked
- 3 seeds: 0, 1, 2
- **Total: 24 experiments**

#### Sequential Execution

```bash
python run_experiments.py --mode sequential
```

#### Generate Shell Script for Parallel Execution

```bash
# Generate run_all.sh
python run_experiments.py --mode generate

# Run sequentially
./run_all.sh

# Or run in parallel (requires GNU parallel)
cat run_all.sh | grep '^python' | parallel -j 4
```

### Visualization

After training completes, generate plots:

```bash
python visualize.py
```

This creates:
- `plots/generalization_gap.png`: Generalization gap over training
- `plots/test_accuracy.png`: Test accuracy over training
- `plots/loss_curves_{arch}.png`: Loss components for masked training
- `plots/lambda_schedule.png`: Lambda warmup schedule

And prints a summary table of final results.

## Training Arguments

Key arguments for `train.py`:

| Argument | Default | Description |
|----------|---------|-------------|
| `--arch` | *required* | Architecture: vgg11/13/16/19 |
| `--mode` | *required* | Training mode: baseline/masked |
| `--seed` | *required* | Random seed |
| `--epochs` | 200 | Number of training epochs |
| `--batch_size` | 128 | Batch size |
| `--lr` | 0.1 | Initial learning rate |
| `--n_masks` | 3 | Number of masks per batch (masked mode) |
| `--lambda_max` | 1.0 | Maximum lambda after warmup (masked mode) |
| `--device` | cuda | Device: cuda/cpu |

## Expected Results

Training 200 epochs on CIFAR-10, we expect:

**Baseline:**
- VGG11: ~91-92% test accuracy
- VGG13: ~92-93% test accuracy
- VGG16: ~93-94% test accuracy
- VGG19: ~93-94% test accuracy

**Masked Loss:**
- Similar or slightly lower test accuracy
- **Lower generalization gap** (train acc - test acc)
- More robust representations

## Implementation Details

### Model Architecture

VGG models adapted for CIFAR-10 (32x32 input):
- Standard VGG convolutional layers with batch normalization
- Adaptive average pooling to 1x1
- Classifier: Linear(512, 512) → ReLU → Linear(512, 10)
- **No dropout** (replaced by neuron masking)

### Training Configuration

- Optimizer: SGD with momentum (0.9)
- Weight decay: 5e-4
- Learning rate schedule: MultiStep decay at 50% and 75% of training (×0.1)
- Data augmentation: RandomCrop(32, padding=4) + RandomHorizontalFlip
- Normalization: CIFAR-10 statistics

### Computational Cost

Masked training performs:
- 1 forward pass for full model
- 3 forward passes for masked models
- **Total: 4× forward passes per batch** (vs baseline)

Estimated training time on single GPU (RTX 3090):
- VGG11: ~2-3 hours (baseline), ~8-10 hours (masked)
- VGG19: ~4-5 hours (baseline), ~15-18 hours (masked)

## Files Generated

After running experiments, you'll have:

```
results/
├── vgg11_baseline_seed0.json
├── vgg11_masked_seed0.json
├── ... (24 total)

checkpoints/
├── vgg11_baseline_seed0.pt
├── vgg11_masked_seed0.pt
├── ... (24 total)

plots/
├── generalization_gap.png
├── test_accuracy.png
├── loss_curves_vgg11.png
├── loss_curves_vgg13.png
├── loss_curves_vgg16.png
├── loss_curves_vgg19.png
└── lambda_schedule.png
```

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{vgg_masked_loss,
  title={VGG Training with Masked Loss Objective},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/loss_fn_ad}
}
```

## License

MIT License

## Contact

For questions or issues, please open an issue on GitHub or contact [your email].
