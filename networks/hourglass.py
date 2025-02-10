import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation=None, use_bias=False):
        super(ConvBlock3D, self).__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=use_bias,
        )
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

class DeconvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, use_bias=False):
        super(DeconvBlock3D, self).__init__()
        self.deconv = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            output_padding=stride - 1,
            bias=use_bias,
        )

    def forward(self, x):
        return self.deconv(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock3D(in_channels, out_channels // 2, kernel_size=1)
        self.bn1 = nn.BatchNorm3d(out_channels // 2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = ConvBlock3D(out_channels // 2, out_channels // 2, kernel_size=3)
        self.bn2 = nn.BatchNorm3d(out_channels // 2)
        self.conv3 = ConvBlock3D(out_channels // 2, out_channels, kernel_size=1)

        if in_channels != out_channels:
            self.shortcut = ConvBlock3D(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        x += residual
        return x


class HourglassModule(nn.Module):
    def __init__(self, depth, num_features):
        super(HourglassModule, self).__init__()
        self.depth = depth
        self.num_features = num_features
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()

    def _build_encoder(self):
        encoder = []
        for _ in range(self.depth):
            encoder.append(ResidualBlock(self.num_features, self.num_features))
            encoder.append(nn.MaxPool3d(kernel_size=2, stride=2))
        return nn.Sequential(*encoder)

    def _build_decoder(self):
        decoder = []
        for _ in range(self.depth - 1):  # One less because the last upsampling is not followed by a downsample
            decoder.append(ResidualBlock(self.num_features, self.num_features))
            decoder.append(nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True))
        # Add the last residual block without a following upsampling
        decoder.append(ResidualBlock(self.num_features, self.num_features))
        return nn.Sequential(*decoder)

    def forward(self, x):
        skip_connections = []

        # Encoder forward pass
        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)

        # Reverse the skip connections for proper indexing in the decoder
        skip_connections = skip_connections[::-1]

        # Decoder forward pass
        for i in range(0, len(self.decoder), 2):
            x = self.decoder[i](x)
            # If not the last layer, add the skip connection and upsample
            if i < len(self.decoder) - 1:
                x += skip_connections[i // 2]
                x = self.decoder[i + 1](x)

        return x

class StackedHourglass(nn.Module):
    def __init__(self, num_stacks=1, depth=3, num_features=64, num_landmarks=6):
        super(StackedHourglass, self).__init__()
        self.num_stacks = num_stacks
        self.depth = depth
        self.num_features = num_features
        self.num_landmarks = num_landmarks

        self.conv1 = ConvBlock3D(1, num_features // 2, kernel_size=5)
        self.bn1 = nn.BatchNorm3d(num_features // 2)
        self.relu = nn.ReLU(inplace=True)
        self.res1 = ResidualBlock(num_features // 2, num_features // 2)
        self.res2 = ResidualBlock(num_features // 2, num_features)

        self.hourglasses = nn.ModuleList([HourglassModule(depth, num_features) for _ in range(num_stacks)])
        self.residuals = nn.ModuleList([ResidualBlock(num_features, num_features) for _ in range(num_stacks)])
        self.conv2 = nn.ModuleList([ConvBlock3D(num_features, num_features, kernel_size=1) for _ in range(num_stacks)])
        self.scores = nn.ModuleList(
            [ConvBlock3D(num_features, num_landmarks, kernel_size=1) for _ in range(num_stacks)])

        self.intermediate_conv = nn.ModuleList(
            [ConvBlock3D(num_landmarks, num_features, kernel_size=1) for _ in range(num_stacks - 1)])

    def forward(self, x):
        x_ori = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.res1(x)
        x = self.res2(x)

        outputs = []
        for i in range(self.num_stacks):
            hg = self.hourglasses[i](x)
            res = self.residuals[i](hg)
            conv2 = self.conv2[i](res)
            score = self.scores[i](conv2)
            outputs.append(score)

            if i < self.num_stacks - 1:
                fc = self.conv2[i](conv2)
                score_ = self.intermediate_conv[i](score)
                x = x + fc + score_
        outputs = F.interpolate(outputs[-1], size=x_ori.shape[2:], mode='trilinear', align_corners=True)

        return outputs


if __name__ == "__main__":
    input_x = torch.zeros((4, 1, 128, 128, 128)).cuda()
    model = StackedHourglass().cuda()
    out_x = model(input_x)
    print(out_x.shape)
