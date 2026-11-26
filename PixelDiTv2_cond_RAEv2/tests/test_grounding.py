"""CPU-only checks for the interface between RAEv2 and PixelDiT2."""
import pytest
import torch
from torch.nn import functional as F

from pixeldit2_c2i import PixelDiT2


def test_h16_shape_and_context_boundary():
    # Inspect the actual H model without allocating its parameter storage.
    with torch.device("meta"):
        net = PixelDiT2()
    assert (net.hidden_size, net.num_heads, net.depth) == (1280, 16, 32)
    assert (net.patch_size, net.in_context_start, net.in_context_len) == (16, 8, 32)
    assert net.grounding_proj.weight.shape == (1280, 1024)


def test_patch_condition_and_repa_features():
    torch.manual_seed(10)
    net = PixelDiT2(hidden_size=16, num_heads=2, depth=9, proj_dropout=0.0).eval()
    x = torch.randn(1, 3, 256, 256)
    t = torch.tensor([0.25])
    y = torch.tensor([42])
    tokens = torch.randn(1, 256, 1024)
    time, label = net.t_embedder(t), net.y_embedder(y).view(1, 1, -1)
    expected = F.silu(F.linear(tokens, net.grounding_proj.weight, net.grounding_proj.bias)
                      + time[:, None] + label)
    torch.testing.assert_close(net._grounded_condition(tokens, time, label), expected)

    block_outputs = []
    handles = [block.register_forward_hook(lambda m, args, out: block_outputs.append(out))
               for block in net.patch_blocks[7:9]]
    try:
        with torch.no_grad():
            rgb, features = net(x, t, y, tokens, repa_align_layer=8)
            assert rgb.shape == x.shape
            assert block_outputs[0].shape == (1, 256, 16)
            assert block_outputs[1].shape == (1, 288, 16)
            torch.testing.assert_close(features, block_outputs[0])
            _, next_features = net(x, t, y, tokens, repa_align_layer=9)
            torch.testing.assert_close(next_features, block_outputs[-1][:, 32:])
    finally:
        for handle in handles:
            handle.remove()
    with pytest.raises(ValueError, match="grounding_tokens"):
        net(x, t, y, tokens[:, :-1])
