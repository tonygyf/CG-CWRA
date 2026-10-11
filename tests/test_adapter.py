import pytest
import torch

from cgcwra.models import CGCWRAAdapter, CGCWRAConfig


def make_adapter() -> CGCWRAAdapter:
    return CGCWRAAdapter(
        CGCWRAConfig(
            hidden_size=16,
            num_heads=4,
            num_layers=2,
            coordinate_hidden_size=8,
            relative_hidden_size=8,
        )
    )


def test_adapter_preserves_shape_and_backpropagates() -> None:
    torch.manual_seed(0)
    adapter = make_adapter()
    features = torch.randn(2, 3, 5, 16, requires_grad=True)
    coordinates = torch.tensor([[0, 0], [0, 1], [1, 0]])

    output = adapter(features, coordinates)

    assert output.shape == features.shape
    output.square().mean().backward()
    assert features.grad is not None
    assert adapter.delta_projection.weight.grad is not None


def test_masks_leave_invalid_patches_and_windows_unchanged() -> None:
    adapter = make_adapter().eval()
    features = torch.randn(1, 3, 4, 16)
    original = features.clone()
    window_mask = torch.tensor([[True, True, False]])
    patch_mask = torch.tensor(
        [[[True, True, False, False], [True, True, True, True], [True, True, True, True]]]
    )

    output = adapter(
        features,
        torch.tensor([[0, 0], [0, 1], [1, 0]]),
        window_mask,
        patch_mask,
    )

    assert torch.equal(output[:, 0, 2:], original[:, 0, 2:])
    assert torch.equal(output[:, 2], original[:, 2])


def test_rejects_bad_coordinate_shape() -> None:
    with pytest.raises(ValueError, match="coordinates"):
        make_adapter()(torch.randn(1, 2, 3, 16), torch.randn(1, 3))
