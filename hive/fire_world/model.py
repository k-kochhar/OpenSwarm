"""
Neural heuristic models for cost-to-go prediction.

V1: HeuristicNet       — simple U-Net (~111K params, baseline)
V2: HeuristicNetV2     — VIN-Attention U-Net (~420K params)
    - U-Net encoder with double convolutions
    - Multi-head self-attention at 8x8 bottleneck
    - Iterative gated decoder (shared weights, T passes)
    - Inspired by VIN, GPPN, and TransPath

Input:  (B, 2, 64, 64) — channel 0: obstacle map, channel 1: goal one-hot
Output: (B, 1, 64, 64) — predicted cost-to-go per cell
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DoubleConv(nn.Module):
    """Two conv-bn-relu blocks with optional dropout."""
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SpatialSelfAttention(nn.Module):
    """
    Multi-head self-attention on a spatial feature map.
    Input:  (B, C, H, W)
    Output: (B, C, H, W)

    Flattens spatial dims into tokens, applies MHSA, reshapes back.
    Includes learnable positional embeddings and a small FFN.
    """
    def __init__(self, dim, num_heads=4, num_tokens=64):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.pos_embed = nn.Parameter(torch.randn(1, num_tokens, dim) * 0.02)

        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        # (B, C, H, W) -> (B, H*W, C)
        tokens = x.flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_embed[:, :H * W, :]

        # Self-attention
        residual = tokens
        qkv = self.qkv(tokens).reshape(B, H * W, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, tokens, head_dim)
        q, k, v = qkv.unbind(0)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, H * W, C)
        out = self.proj(out)
        tokens = self.norm1(residual + out)

        # FFN
        residual = tokens
        tokens = self.norm2(residual + self.ffn(tokens))

        # (B, H*W, C) -> (B, C, H, W)
        return tokens.transpose(1, 2).reshape(B, C, H, W)


# ---------------------------------------------------------------------------
# V1: Original simple U-Net (kept for backward compat)
# ---------------------------------------------------------------------------

class HeuristicNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = ConvBnRelu(2, 16)
        self.e2 = ConvBnRelu(16, 32)
        self.e3 = ConvBnRelu(32, 64)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBnRelu(64, 64)
        self.up3 = ConvBnRelu(64 + 64, 32)
        self.up2 = ConvBnRelu(32 + 32, 16)
        self.up1 = ConvBnRelu(16 + 16, 16)
        self.out_conv = nn.Conv2d(16, 1, 1)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.up3(torch.cat([self.upsample(b), e3], dim=1))
        d2 = self.up2(torch.cat([self.upsample(d3), e2], dim=1))
        d1 = self.up1(torch.cat([self.upsample(d2), e1], dim=1))
        return self.out_conv(d1)


# ---------------------------------------------------------------------------
# V2: VIN-Attention U-Net with iterative gated decoder
# ---------------------------------------------------------------------------

class HeuristicNetV2(nn.Module):
    """
    U-Net encoder + spatial self-attention + iterative gated decoder.

    Architecture:
      Encoder:  3 stages (24→48→96), double convolutions, MaxPool
      Bottleneck: Conv + Multi-Head Self-Attention (4 heads, 8x8=64 tokens)
      Decoder:  3 stages with skip connections (shared weights across T iterations)
                Gated residual update inspired by GPPN
    """

    def __init__(self, num_iterations=4, dropout=0.1):
        super().__init__()
        self.num_iterations = num_iterations
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        # === Encoder (runs once) ===
        self.e1 = DoubleConv(2, 24)
        self.e2 = DoubleConv(24, 48)
        self.e3 = DoubleConv(48, 96)

        # === Bottleneck with self-attention ===
        self.bn_conv1 = ConvBnRelu(96, 96)
        self.bn_attn = SpatialSelfAttention(dim=96, num_heads=4, num_tokens=64)
        self.bn_conv2 = ConvBnRelu(96, 96)

        # === Iterative gated decoder (shared weights across T iterations) ===
        # Gate: condition bottleneck on previous estimate
        self.iter_gate = nn.Sequential(
            nn.Conv2d(96 + 1, 96, 1),
            nn.Sigmoid(),
        )

        # Decoder stages
        self.d3 = DoubleConv(96 + 96, 48, dropout=dropout)
        self.d2 = DoubleConv(48 + 48, 24, dropout=dropout)
        self.d1 = DoubleConv(24 + 24, 24, dropout=dropout)

        # Output head
        self.out_conv = nn.Conv2d(24, 1, 1)

        # Update gate (GPPN-inspired): decides how much to update vs keep
        self.update_gate = nn.Sequential(
            nn.Conv2d(24 + 1, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, return_intermediates=False):
        B = x.shape[0]

        # === Encoder ===
        e1 = self.e1(x)                    # (B, 24, 64, 64)
        e2 = self.e2(self.pool(e1))         # (B, 48, 32, 32)
        e3 = self.e3(self.pool(e2))         # (B, 96, 16, 16)

        # === Bottleneck with attention ===
        b = self.bn_conv1(self.pool(e3))    # (B, 96, 8, 8)
        b = b + self.bn_attn(b)            # residual attention
        b = self.bn_conv2(b)               # (B, 96, 8, 8)

        # === Iterative decoder ===
        v_prev = torch.zeros(B, 1, x.shape[2], x.shape[3], device=x.device)
        intermediates = []

        for t in range(self.num_iterations):
            # Gate the bottleneck using previous cost-to-go estimate
            v_down = F.avg_pool2d(F.avg_pool2d(F.avg_pool2d(v_prev, 2), 2), 2)  # (B,1,8,8)
            gate = self.iter_gate(torch.cat([b, v_down], dim=1))  # (B, 96, 8, 8)
            b_gated = b * gate

            # Decode
            d3 = self.d3(torch.cat([self.upsample(b_gated), e3], dim=1))  # (B, 48, 16, 16)
            d2 = self.d2(torch.cat([self.upsample(d3), e2], dim=1))       # (B, 24, 32, 32)
            d1 = self.d1(torch.cat([self.upsample(d2), e1], dim=1))       # (B, 24, 64, 64)

            # Output
            v_raw = self.out_conv(d1)  # (B, 1, 64, 64)

            # Gated residual update
            ug = self.update_gate(torch.cat([d1, v_prev], dim=1))  # (B, 1, 64, 64)
            v_new = ug * v_raw + (1 - ug) * v_prev

            v_prev = v_new
            if return_intermediates:
                intermediates.append(v_new)

        if return_intermediates:
            return v_new, intermediates
        return v_new


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=== V1: HeuristicNet ===")
    net1 = HeuristicNet()
    print(f"Parameters: {count_parameters(net1):,}")
    x = torch.randn(4, 2, 64, 64)
    y = net1(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")

    print("\n=== V2: HeuristicNetV2 ===")
    net2 = HeuristicNetV2(num_iterations=4)
    print(f"Parameters: {count_parameters(net2):,}")
    y2, intermediates = net2(x, return_intermediates=True)
    print(f"Input: {x.shape} -> Output: {y2.shape}")
    print(f"Intermediate outputs: {len(intermediates)}")
