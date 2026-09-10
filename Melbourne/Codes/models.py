'''
A dual-stream model learns SAR and optical features separately first,
then fuses them in the decoder. This is preferable to treating
all 12 bands identically, because Sentinel-1 radar and Sentinel-2
reflectance have very different value distributions and meanings.
A Dual Stream U-Net was specifically developed and evaluated 
for Sentinel-1/Sentinel-2 fusion with OSCD-style urban change detection.
DS-UNet project and paper
We can train both early-fusion and double stream models to see their results :) .

'''
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# BASIC BLOCKS
# ============================================================

def get_group_count(channels, maximum_groups=8):
    groups = min(channels, maximum_groups)

    while channels % groups != 0:
        groups -= 1

    return groups


class ConvNormActivation(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                get_group_count(out_channels),
                out_channels,
            ),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            ConvNormActivation(in_channels, out_channels),
            ConvNormActivation(out_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )

        self.skip_projection = nn.Conv2d(
            skip_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )

        self.fusion = DoubleConv(
            out_channels * 2,
            out_channels,
        )

    def forward(self, x, skip):
        x = self.up(x)

        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        skip = self.skip_projection(skip)

        x = torch.cat([x, skip], dim=1)

        return self.fusion(x)


# ============================================================
# TEMPORAL FEATURE BLOCK
# ============================================================
#
# Uses:
#   t1
#   t2
#   absolute difference: |t2 - t1|
#   interaction: t1 * t2
#
# This makes the network explicitly aware of temporal change.
# ============================================================

class TemporalContrastStem(nn.Module):

    def __init__(self, channels_per_time, out_channels):
        super().__init__()

        self.channels_per_time = channels_per_time

        self.projection = DoubleConv(
            channels_per_time * 4,
            out_channels,
        )

    def forward(self, x):
        t1 = x[:, :self.channels_per_time]
        t2 = x[:, self.channels_per_time:]

        difference = torch.abs(t2 - t1)
        interaction = t1 * t2

        temporal_features = torch.cat(
            [t1, t2, difference, interaction],
            dim=1,
        )

        return self.projection(temporal_features)


# ============================================================
# EARLY-FUSION MODEL
# ============================================================
#
# Input channels:
#
# 0-1:   S1 T1 VV, VH
# 2-5:   S2 T1 B2, B3, B4, B8
# 6-7:   S1 T2 VV, VH
# 8-11:  S2 T2 B2, B3, B4, B8
#
# Input shape:  (B, 12, H, W)
# Output shape: (B, 1, H, W)
# ============================================================

class ContrastPyramidEarlyFusionUNet(nn.Module):

    def __init__(
        self,
        in_channels=12,
        base_channels=32,
        out_channels=1,
    ):
        super().__init__()

        if in_channels != 12:
            raise ValueError(
                "ContrastPyramidEarlyFusionUNet expects 12 input channels."
            )

        # Each time step contains:
        # S1: 2 bands + S2: 4 bands = 6 channels.
        self.temporal_stem = TemporalContrastStem(
            channels_per_time=6,
            out_channels=base_channels,
        )

        self.encoder1 = DoubleConv(
            base_channels,
            base_channels,
        )

        self.encoder2 = DownBlock(
            base_channels,
            base_channels * 2,
        )

        self.encoder3 = DownBlock(
            base_channels * 2,
            base_channels * 4,
        )

        self.encoder4 = DownBlock(
            base_channels * 4,
            base_channels * 8,
        )

        self.bottleneck = DownBlock(
            base_channels * 8,
            base_channels * 16,
        )

        self.decoder4 = UpBlock(
            base_channels * 16,
            base_channels * 8,
            base_channels * 8,
        )

        self.decoder3 = UpBlock(
            base_channels * 8,
            base_channels * 4,
            base_channels * 4,
        )

        self.decoder2 = UpBlock(
            base_channels * 4,
            base_channels * 2,
            base_channels * 2,
        )

        self.decoder1 = UpBlock(
            base_channels * 2,
            base_channels,
            base_channels,
        )

        self.output_head = nn.Conv2d(
            base_channels,
            out_channels,
            kernel_size=1,
        )

    def forward(self, image):
        x = self.temporal_stem(image)

        encoder1 = self.encoder1(x)
        encoder2 = self.encoder2(encoder1)
        encoder3 = self.encoder3(encoder2)
        encoder4 = self.encoder4(encoder3)

        bottleneck = self.bottleneck(encoder4)

        decoder4 = self.decoder4(bottleneck, encoder4)
        decoder3 = self.decoder3(decoder4, encoder3)
        decoder2 = self.decoder2(decoder3, encoder2)
        decoder1 = self.decoder1(decoder2, encoder1)

        return self.output_head(decoder1)


# ============================================================
# CROSS-MODAL GATED FUSION
# ============================================================
#
# Learns how much useful information comes from:
# - Sentinel-1 radar features
# - Sentinel-2 optical features
# - their feature difference
# ============================================================
class CrossModalGatedFusion(nn.Module):

    def __init__(self, channels):
        super().__init__()

        # Input to gates: concatenated S1 + S2 features
        self.s1_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1), nn.Sigmoid()
        )

        self.s2_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1), nn.Sigmoid()
        )

        # Updated channels: weighted_s1 (C) + weighted_s2 (C) + s1_diff (C/2) + s2_diff (C/2) = 3 * channels
        self.fusion = DoubleConv(channels * 3, channels)

    def forward(self, sentinel1_features, sentinel2_features):
        # 1. Joint Cross-Modal Attention Gates
        joint_features = torch.cat(
            [sentinel1_features, sentinel2_features], dim=1
        )

        s1_weight = self.s1_gate(joint_features)
        s2_weight = self.s2_gate(joint_features)

        weighted_s1 = sentinel1_features * s1_weight
        weighted_s2 = sentinel2_features * s2_weight

        # 2. Extract Temporal Pairs (assuming features contain [t1_features, t2_features])
        # Split along channel dimension into Time 1 and Time 2
        s1_t1, s1_t2 = torch.chunk(sentinel1_features, chunks=2, dim=1)
        s2_t1, s2_t2 = torch.chunk(sentinel2_features, chunks=2, dim=1)

        # 3. Calculate TEMPORAL Differences (T2 - T1 per modality)
        s1_temp_diff = torch.abs(s1_t2 - s1_t1)
        s2_temp_diff = torch.abs(s2_t2 - s2_t1)

        # 4. Fuse Weighted Features + Temporal Change Descriptors
        fusion_input = torch.cat(
            [weighted_s1, weighted_s2, s1_temp_diff, s2_temp_diff], dim=1
        )

        return self.fusion(fusion_input)


# ============================================================
# DUAL-STREAM MODEL
# ============================================================
#
# Sentinel-1 input:
# Channel 0: S1 T1 VV
# Channel 1: S1 T1 VH
# Channel 2: S1 T2 VV
# Channel 3: S1 T2 VH
#
# Sentinel-2 input:
# Channels 0-3: S2 T1 B2, B3, B4, B8
# Channels 4-7: S2 T2 B2, B3, B4, B8
#
# Output:
# Binary-change logits, shape (B, 1, H, W)
# ============================================================

class CrossModalTemporalGateUNet(nn.Module):

    def __init__(
        self,
        s1_in_channels=4,
        s2_in_channels=8,
        base_channels=32,
        out_channels=1,
    ):
        super().__init__()

        if s1_in_channels % 2 != 0:
            raise ValueError("S1 input channels must contain T1 and T2.")

        if s2_in_channels % 2 != 0:
            raise ValueError("S2 input channels must contain T1 and T2.")

        self.s1_temporal_stem = TemporalContrastStem(
            channels_per_time=s1_in_channels // 2,
            out_channels=base_channels,
        )

        self.s2_temporal_stem = TemporalContrastStem(
            channels_per_time=s2_in_channels // 2,
            out_channels=base_channels,
        )

        self.s1_encoder1 = DoubleConv(
            base_channels,
            base_channels,
        )

        self.s1_encoder2 = DownBlock(
            base_channels,
            base_channels * 2,
        )

        self.s1_encoder3 = DownBlock(
            base_channels * 2,
            base_channels * 4,
        )

        self.s1_encoder4 = DownBlock(
            base_channels * 4,
            base_channels * 8,
        )

        self.s1_bottleneck = DownBlock(
            base_channels * 8,
            base_channels * 16,
        )

        self.s2_encoder1 = DoubleConv(
            base_channels,
            base_channels,
        )

        self.s2_encoder2 = DownBlock(
            base_channels,
            base_channels * 2,
        )

        self.s2_encoder3 = DownBlock(
            base_channels * 2,
            base_channels * 4,
        )

        self.s2_encoder4 = DownBlock(
            base_channels * 4,
            base_channels * 8,
        )

        self.s2_bottleneck = DownBlock(
            base_channels * 8,
            base_channels * 16,
        )

        self.fusion1 = CrossModalGatedFusion(base_channels)
        self.fusion2 = CrossModalGatedFusion(base_channels * 2)
        self.fusion3 = CrossModalGatedFusion(base_channels * 4)
        self.fusion4 = CrossModalGatedFusion(base_channels * 8)
        self.fusion5 = CrossModalGatedFusion(base_channels * 16)

        self.decoder4 = UpBlock(
            base_channels * 16,
            base_channels * 8,
            base_channels * 8,
        )

        self.decoder3 = UpBlock(
            base_channels * 8,
            base_channels * 4,
            base_channels * 4,
        )

        self.decoder2 = UpBlock(
            base_channels * 4,
            base_channels * 2,
            base_channels * 2,
        )

        self.decoder1 = UpBlock(
            base_channels * 2,
            base_channels,
            base_channels,
        )

        self.output_head = nn.Conv2d(
            base_channels,
            out_channels,
            kernel_size=1,
        )

    def forward(self, sentinel1, sentinel2):
        # Temporal feature extraction for each sensor.
        s1 = self.s1_temporal_stem(sentinel1)
        s2 = self.s2_temporal_stem(sentinel2)

        # First encoder scale.
        s1_encoder1 = self.s1_encoder1(s1)
        s2_encoder1 = self.s2_encoder1(s2)
        fusion1 = self.fusion1(s1_encoder1, s2_encoder1)

        # Second encoder scale.
        s1_encoder2 = self.s1_encoder2(s1_encoder1)
        s2_encoder2 = self.s2_encoder2(s2_encoder1)
        fusion2 = self.fusion2(s1_encoder2, s2_encoder2)

        # Third encoder scale.
        s1_encoder3 = self.s1_encoder3(s1_encoder2)
        s2_encoder3 = self.s2_encoder3(s2_encoder2)
        fusion3 = self.fusion3(s1_encoder3, s2_encoder3)

        # Fourth encoder scale.
        s1_encoder4 = self.s1_encoder4(s1_encoder3)
        s2_encoder4 = self.s2_encoder4(s2_encoder3)
        fusion4 = self.fusion4(s1_encoder4, s2_encoder4)

        # Deepest features.
        s1_bottleneck = self.s1_bottleneck(s1_encoder4)
        s2_bottleneck = self.s2_bottleneck(s2_encoder4)
        fusion5 = self.fusion5(s1_bottleneck, s2_bottleneck)

        # Decoder uses fused multi-scale features.
        decoder4 = self.decoder4(fusion5, fusion4)
        decoder3 = self.decoder3(decoder4, fusion3)
        decoder2 = self.decoder2(decoder3, fusion2)
        decoder1 = self.decoder1(decoder2, fusion1)

        return self.output_head(decoder1)


# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__":
    batch_size = 2
    height = 256
    width = 256

    # Early-fusion model
    early_fusion_model = ContrastPyramidEarlyFusionUNet()

    stacked_image = torch.randn(
        batch_size,
        12,
        height,
        width,
    )

    early_fusion_output = early_fusion_model(stacked_image)

    print("Early-fusion output shape:")
    print(early_fusion_output.shape)

    # Dual-stream model
    dual_stream_model = CrossModalTemporalGateUNet()

    sentinel1 = torch.randn(
        batch_size,
        4,
        height,
        width,
    )

    sentinel2 = torch.randn(
        batch_size,
        8,
        height,
        width,
    )

    dual_stream_output = dual_stream_model(
        sentinel1,
        sentinel2,
    )

    print("Dual-stream output shape:")
    print(dual_stream_output.shape)
