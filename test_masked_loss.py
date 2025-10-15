"""Quick test script to verify masked loss implementation."""

import torch
import torch.nn as nn
from models.vgg_cifar import VGG11
from train import (
    generate_random_neuron_masks,
    NeuronMaskingHook,
    compute_loss_with_masking,
    compute_masked_loss_objective,
    get_lambda_warmup
)


def test_mask_generation():
    """Test random neuron mask generation."""
    print("Testing mask generation...")

    n_neurons = 512
    n_masks = 5

    masks = generate_random_neuron_masks(n_neurons, n_masks)

    assert len(masks) == n_masks, "Wrong number of masks"

    for i, mask in enumerate(masks):
        assert mask.shape == (n_neurons,), f"Mask {i} has wrong shape"
        assert mask.dtype == bool, f"Mask {i} has wrong dtype"
        n_active = mask.sum()
        assert 1 <= n_active <= n_neurons - 2, f"Mask {i} has {n_active} active neurons"

    print(f"✓ Generated {n_masks} masks, each with shape ({n_neurons},)")
    print(f"  Active neurons per mask: {[m.sum() for m in masks]}")


def test_masking_hook():
    """Test that masking hook correctly zeros out neurons."""
    print("\nTesting masking hook...")

    batch_size = 4
    n_neurons = 512

    # Create dummy activation
    activation = torch.randn(batch_size, n_neurons)

    # Create mask that keeps only first 100 neurons
    mask = torch.zeros(n_neurons, dtype=bool)
    mask[:100] = True

    # Apply hook
    hook = NeuronMaskingHook(mask.numpy())

    # Simulate forward hook call
    masked_activation = hook(None, None, activation)

    # Check that inactive neurons are zeroed
    assert torch.allclose(masked_activation[:, 100:], torch.zeros_like(masked_activation[:, 100:])), \
        "Inactive neurons should be zero"

    # Check that active neurons are preserved
    assert torch.allclose(masked_activation[:, :100], activation[:, :100]), \
        "Active neurons should be preserved"

    print("✓ Masking hook correctly zeros out inactive neurons")


def test_masked_loss_computation():
    """Test masked loss computation."""
    print("\nTesting masked loss computation...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    model = VGG11(num_classes=10, use_batchnorm=True).to(device)
    model.eval()

    # Create dummy data
    batch_size = 8
    inputs = torch.randn(batch_size, 3, 32, 32).to(device)
    targets = torch.randint(0, 10, (batch_size,)).to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Get masking layer
    masking_layer = model.get_masking_layer()

    # Compute losses
    with torch.no_grad():
        loss_full, loss_masked_avg = compute_loss_with_masking(
            model, inputs, targets, criterion, masking_layer, n_masks=3
        )

    print(f"✓ Loss computation successful")
    print(f"  L_full: {loss_full.item():.4f}")
    print(f"  L_masked_avg: {loss_masked_avg.item():.4f}")

    # Loss should be positive
    assert loss_full > 0, "Full loss should be positive"
    assert loss_masked_avg > 0, "Masked loss should be positive"

    # Masked loss should generally be higher (model performs worse when masked)
    print(f"  Masked loss is {'higher' if loss_masked_avg > loss_full else 'lower'} than full loss")


def test_loss_objective():
    """Test masked loss objective computation."""
    print("\nTesting loss objective computation...")

    loss_full = torch.tensor(2.0)
    loss_masked_avg = torch.tensor(1.5)  # Below clamp threshold
    lambda_current = 0.5

    # Compute objective (no clamping needed)
    loss_total = compute_masked_loss_objective(loss_full, loss_masked_avg, lambda_current)

    # masked_avg (1.5) > 0.5 * loss_full (1.0), so clamping to 1.0
    # L_total = 2.0 - 0.5 * min(1.5, 1.0) = 2.0 - 0.5 * 1.0 = 1.5
    expected = loss_full - lambda_current * torch.min(loss_masked_avg, 0.5 * loss_full)
    assert torch.allclose(loss_total, expected), f"Expected {expected}, got {loss_total}"

    print(f"✓ Loss objective: L_total = {loss_total.item():.4f}")

    # Test clamping when masked loss is too high
    loss_masked_avg = torch.tensor(5.0)  # Much higher than full loss
    loss_total = compute_masked_loss_objective(loss_full, loss_masked_avg, lambda_current)

    # Should clamp to 0.5 * loss_full = 1.0
    # So: L_total = 2.0 - 0.5 * min(5.0, 1.0) = 2.0 - 0.5 * 1.0 = 1.5
    expected = loss_full - lambda_current * (0.5 * loss_full)
    assert torch.allclose(loss_total, expected), f"Clamping failed: expected {expected}, got {loss_total}"

    print(f"✓ Clamping works: L_total = {loss_total.item():.4f} (with high masked loss)")


def test_lambda_warmup():
    """Test lambda warmup schedule."""
    print("\nTesting lambda warmup schedule...")

    total_epochs = 200
    warmup_epochs = int(0.2 * total_epochs)  # 40

    # Test at various epochs
    test_epochs = [1, 20, 40, 100, 200]

    for epoch in test_epochs:
        lambda_val = get_lambda_warmup(epoch, total_epochs, lambda_max=1.0)

        if epoch <= warmup_epochs:
            expected = epoch / warmup_epochs
        else:
            expected = 1.0

        assert abs(lambda_val - expected) < 1e-6, f"Epoch {epoch}: expected {expected}, got {lambda_val}"

        print(f"  Epoch {epoch:3d}: λ = {lambda_val:.4f}")

    print("✓ Lambda warmup schedule is correct")


def main():
    print("="*70)
    print("Testing Masked Loss Implementation")
    print("="*70)

    test_mask_generation()
    test_masking_hook()
    test_masked_loss_computation()
    test_loss_objective()
    test_lambda_warmup()

    print("\n" + "="*70)
    print("All tests passed! ✓")
    print("="*70)


if __name__ == '__main__':
    main()
