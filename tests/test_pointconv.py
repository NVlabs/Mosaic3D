import pytest
import torch
import warp
from warp.convnet.geometry.ops.neighbor_search_continuous import NeighborSearchArgs
from warp.convnet.geometry.ops.point_pool import FeaturePoolingArgs
from warp.convnet.geometry.point_collection import PointCollection
from warp.convnet.models.point_conv_unet import PointConvUNet


@pytest.mark.slow
def test_instantiation():
    warp.init()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = 4
    C_in, C_out = 3, 7
    N_min, N_max = 10000, 20000
    Ns = torch.randint(N_min, N_max, (B,))
    coords = [torch.rand((N, 3)) for N in Ns]
    features = [torch.rand((N, C_in)) for N in Ns]
    pc = PointCollection(coords, features).to(device)

    model = PointConvUNet(
        in_channels=C_in,
        out_channels=C_out,
        down_channels=[16, 32, 64, 128],
        up_channels=[16, 32, 64, 128],
        downsample_voxel_size=0.02,
        neighbor_search_args=NeighborSearchArgs("radius", 0.01),
        pooling_args=FeaturePoolingArgs(
            pooling_mode="reductions",
        ),
        num_levels=3,
    ).to(device)

    feature_maps = model(pc)
    out = feature_maps[0]

    assert out.feature_shape == (Ns.sum().item(), C_out)
