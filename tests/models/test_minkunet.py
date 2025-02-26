import pytest
import torch
import warp
from src.models.networks.warp.mink_unet import MinkUNetToCLIP


# Check if CUDA is available
CUDA_AVAILABLE = torch.cuda.is_available()
skip_if_no_cuda = pytest.mark.skipif(not CUDA_AVAILABLE, reason="Test requires CUDA device")


@pytest.fixture(scope="session", autouse=True)
def setup_warp():
    """Initialize warp before running any tests."""
    warp.init()


@pytest.fixture
def device():
    """Return the CUDA device."""
    return torch.device("cuda")


@pytest.fixture
def default_backbone_cfg():
    return {
        "in_channels": 3,  # Assuming RGB input
        "out_channels": 512,  # Common CLIP dimension
        "planes": [32, 64, 128, 256, 128, 128, 96, 96],
        "layers": [2, 2, 2, 2, 2, 2, 2, 2],
        "init_dim": 32,
        "voxel_size": 0.02,
        "block_types": "bbbbbbbb",
    }


@skip_if_no_cuda
def test_minkunet_initialization(default_backbone_cfg, device):
    """Test basic initialization of MinkUNetToCLIP."""
    model = MinkUNetToCLIP(
        backbone_cfg=default_backbone_cfg,
        adapter_cfg=None,  # Not using adapter as per config
        voxel_size=0.02,
    ).to(device)

    assert model is not None
    assert model.voxel_size == 0.02
    assert model.backbone_3d is not None
    assert model.adapter is None


@skip_if_no_cuda
def test_minkunet_forward(device):
    """Test forward pass of MinkUNetToCLIP with dummy data."""
    # Create minimal model
    backbone_cfg = {
        "in_channels": 3,
        "out_channels": 512,
        "planes": [32, 64, 128, 256, 128, 128, 96, 96],
        "layers": [2, 2, 2, 2, 2, 2, 2, 2],
        "init_dim": 32,
        "voxel_size": 0.02,
        "block_types": "bbbbbbbb",
    }

    model = MinkUNetToCLIP(backbone_cfg=backbone_cfg, voxel_size=0.02).to(device)

    # Create dummy input data
    batch_size = 2
    num_points = 100

    data_dict = {
        "feat": torch.randn(num_points * batch_size, 3, device=device),
        "coord": torch.randn(num_points * batch_size, 3, device=device),
        "offset": torch.tensor([num_points, num_points * 2], dtype=torch.int32, device=device),
    }

    # Run forward pass
    with torch.no_grad():
        output = model(data_dict)

    # Check outputs
    assert "clip_feat" in output
    assert "pc" in output
    assert output["clip_feat"].shape[-1] == backbone_cfg["out_channels"]
    assert len(output["clip_feat"]) == num_points * batch_size
