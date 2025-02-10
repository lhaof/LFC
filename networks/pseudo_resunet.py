import numpy as np
import torch
import torch.nn as nn

from networks.pseudo_module import find_external_cube_corners


def generate_gaussian_heatmap(a_shape, points, sigma=5.0):
    batch_size, numbers, x, y, z = a_shape
    grid_x, grid_y, grid_z = torch.meshgrid(torch.arange(x), torch.arange(y), torch.arange(z))
    heatmap = torch.zeros((batch_size, numbers, x, y, z))

    for n in range(numbers):
        point = points[n]
        distance = (grid_x - point[0]) ** 2 + (grid_y - point[1]) ** 2 + (grid_z - point[2]) ** 2
        heatmap[0, n] = torch.exp(-distance / (2 * sigma ** 2))
    heatmap = 255 * (heatmap - torch.min(heatmap)) / (torch.max(heatmap) - torch.min(heatmap))

    return heatmap

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding."""
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
            self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def make_res_layer(inplanes, planes, blocks, stride=1):
    downsample = nn.Sequential(
        conv1x1(inplanes, planes, stride),
        nn.BatchNorm3d(planes),
    )

    layers = []
    layers.append(BasicBlock(inplanes, planes, stride, downsample))
    for _ in range(1, blocks):
        layers.append(BasicBlock(planes, planes))

    return nn.Sequential(*layers)


class DoubleConv(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=int(kernel_size / 2)),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, dilation=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        return self.conv(input)


class SingleConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(SingleConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv3d(in_ch, out_ch, 3, padding=1), nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True))

    def forward(self, input):
        return self.conv(input)


class ResUnetPseudo_3d(nn.Module):
    def __init__(self, in_ch=1, channels=32, blocks=3):
        super(ResUnetPseudo_3d, self).__init__()

        self.in_conv = DoubleConv(in_ch, channels, stride=2, kernel_size=3)
        self.layer1 = make_res_layer(channels, channels * 2, blocks, stride=2)
        self.layer2 = make_res_layer(channels * 2, channels * 4, blocks, stride=2)
        self.layer3 = make_res_layer(channels * 4, channels * 8, blocks, stride=2)

        self.up5 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv5 = DoubleConv(channels * 12, channels * 4)
        self.up6 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv6 = DoubleConv(channels * 6, channels * 2)
        self.up7 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv7 = DoubleConv(channels * 3, channels)
        self.up8 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv8 = DoubleConv(channels, 12)

        self.up5p = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv5p = DoubleConv(channels * 12, channels * 4)
        self.up6p = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv6p = DoubleConv(channels * 6, channels * 2)
        self.up7p = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv7p = DoubleConv(channels * 3, channels)
        self.up8p = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv8p = DoubleConv(channels, 12)

        self.up5po = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv5po = DoubleConv(channels * 12, channels * 4)
        self.up6po = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv6po = DoubleConv(channels * 6, channels * 2)
        self.up7po = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv7po = DoubleConv(channels * 3, channels)
        self.up8po = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv8po = DoubleConv(channels, 12)

    def forward(self, input):
        c1 = self.in_conv(input)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)

        up_5 = self.up5(c4)
        merge5 = torch.cat([up_5, c3], dim=1)
        c5 = self.conv5(merge5)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c2], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c1], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        c8 = self.conv8(up_8)

        up_5p = self.up5p(c4)
        merge5p = torch.cat([up_5p, c3], dim=1)
        c5p = self.conv5p(merge5p)
        up_6p = self.up6p(c5p)
        merge6p = torch.cat([up_6p, c2], dim=1)
        c6p = self.conv6p(merge6p)
        up_7p = self.up7p(c6p)
        merge7p = torch.cat([up_7p, c1], dim=1)
        c7p = self.conv7p(merge7p)
        up_8p = self.up8p(c7p)
        c8p = self.conv8p(up_8p)

        up_5po = self.up5po(c4)
        merge5po = torch.cat([up_5po, c3], dim=1)
        c5po = self.conv5po(merge5po)
        up_6po = self.up6po(c5po)
        merge6po = torch.cat([up_6po, c2], dim=1)
        c6po = self.conv6po(merge6po)
        up_7po = self.up7po(c6po)
        merge7po = torch.cat([up_7po, c1], dim=1)
        c7po = self.conv7p(merge7po)
        up_8po = self.up8p(c7po)
        c8po = self.conv8p(up_8po)

        # for line_idx in range(12):
        # small_brain = np.asarray(c8p.detach().cpu())
        # import cc3d
        # for bs in range(small_brain.shape[0]):
        #     small_brain_i = small_brain[bs][0]
        #     cc_seg, max_label_count = cc3d.connected_components((small_brain_i > 1), return_N=True)
        #     # print('number of labels', max_label_count)
        #     # print('max_label', np.max(cc_seg))
        #     # print('ccseg', cc_seg.shape)
        #     cc_seg[cc_seg != 1] = 0
        #     corners = find_external_cube_corners(cc_seg)
        return c8, c8p, c8po

class ResUnetSeg(nn.Module):
    def __init__(self, in_ch=1, channels=32, blocks=3):
        super(ResUnetSeg, self).__init__()
        self.in_convp = DoubleConv(in_ch, channels, stride=2, kernel_size=3)
        self.layer1p = make_res_layer(channels, channels * 2, blocks, stride=2)
        self.layer2p = make_res_layer(channels * 2, channels * 4, blocks, stride=2)
        self.layer3p = make_res_layer(channels * 4, channels * 8, blocks, stride=2)

        self.up5p = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv5p = DoubleConv(channels * 12, channels * 4)
        self.up6p = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv6p = DoubleConv(channels * 6, channels * 2)
        self.up7p = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv7p = DoubleConv(channels * 3, channels)
        self.up8p = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv8p = DoubleConv(channels, 1)

    def forward(self, input):
        c1p = self.in_convp(input)
        c2p = self.layer1p(c1p)
        c3p = self.layer2p(c2p)
        c4p = self.layer3p(c3p)

        up_5p = self.up5p(c4p)
        merge5p = torch.cat([up_5p, c3p], dim=1)
        c5p = self.conv5p(merge5p)
        up_6p = self.up6p(c5p)
        merge6p = torch.cat([up_6p, c2p], dim=1)
        c6p = self.conv6p(merge6p)
        up_7p = self.up7p(c6p)
        merge7p = torch.cat([up_7p, c1p], dim=1)
        c7p = self.conv7p(merge7p)
        up_8p = self.up8p(c7p)
        c8p = self.conv8p(up_8p)

        small_brain = np.asarray(c8p.detach().cpu())
        import cc3d
        for bs in range(small_brain.shape[0]):
            small_brain_i = small_brain[bs][0]
            cc_seg, max_label_count = cc3d.connected_components((small_brain_i > 1), return_N=True)
            # print('number of labels', max_label_count)
            # print('max_label', np.max(cc_seg))
            # print('ccseg', cc_seg.shape)
            cc_seg[cc_seg != 1] = 0
            corners = find_external_cube_corners(cc_seg)

        return c8p


class ResUnetPseudo(nn.Module):
    def __init__(self, in_ch=1, channels=32, blocks=3, out_ch=6, out_pseudo=1):
        super(ResUnetPseudo, self).__init__()

        self.in_conv = DoubleConv(in_ch, channels, stride=2, kernel_size=3)
        self.layer1 = make_res_layer(channels, channels * 2, blocks, stride=2)
        self.layer2 = make_res_layer(channels * 2, channels * 4, blocks, stride=2)
        self.layer3 = make_res_layer(channels * 4, channels * 8, blocks, stride=2)

        self.up5 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv5 = DoubleConv(channels * 12, channels * 4)
        self.up6 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv6 = DoubleConv(channels * 6, channels * 2)
        self.up7 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv7 = DoubleConv(channels * 3, channels)
        self.up8 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv8 = DoubleConv(channels, out_ch)

        self.up5p = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv5p = DoubleConv(channels * 12, channels * 4)
        self.up6p = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv6p = DoubleConv(channels * 6, channels * 2)
        self.up7p = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv7p = DoubleConv(channels * 3, channels)
        self.up8p = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv8p = DoubleConv(channels, out_pseudo)

        # self.fusion_layers = [DoubleConv(2, 1).cuda() for i in range(12)]

    def forward(self, input):
        c1 = self.in_conv(input)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)

        up_5 = self.up5(c4)
        merge5 = torch.cat([up_5, c3], dim=1)
        c5 = self.conv5(merge5)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c2], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c1], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        c8 = self.conv8(up_8)

        #
        # up_5p = self.up5p(c4)
        # merge5p = torch.cat([up_5p, c3], dim=1)
        # c5p = self.conv5p(merge5p)
        # up_6p = self.up6p(c5p)
        # up_6p = self.conv6pp(up_6p)
        # up_7p = self.up7p(up_6p)
        # up_7p = self.conv7pp(up_7p)
        # up_8p = self.up8p(up_7p)
        # c8p = self.conv8pp(up_8p)

        # skip unet for prediction
        up_5p = self.up5p(c4)
        merge5p = torch.cat([up_5p, c3], dim=1)
        c5p = self.conv5p(merge5p)
        up_6p = self.up6p(c5p)
        merge6p = torch.cat([up_6p, c2], dim=1)
        c6p = self.conv6p(merge6p)
        up_7p = self.up7p(c6p)
        merge7p = torch.cat([up_7p, c1], dim=1)
        c7p = self.conv7p(merge7p)
        up_8p = self.up8p(c7p)
        c8p = self.conv8p(up_8p)

        # c8p [batch_size, channel, h, w, d]  # small, big, bone
        # brain = np.asarray(c8p.detach().cpu())
        # import cc3d
        # for bs in range(brain.shape[0]):
        #     cc_seg, max_label_count = cc3d.connected_components((brain[bs][0] > 1), return_N=True)
        #     cc_seg[cc_seg != 1] = 0
        #     # corners_small = find_external_cube_corners(cc_seg, flag='small')
        #     corners_small = get_end_points(cc_seg, flag='small')
        #     # print(corners_small)
        #
        #     cc_seg, max_label_count = cc3d.connected_components((brain[bs][1] > 1), return_N=True)
        #     cc_seg[cc_seg != 1] = 0
        #     # corners_big = find_external_cube_corners(cc_seg, flag='big')[:4]
        #     corners_big = get_end_points(cc_seg, flag='big')[:4]
        #
        #     cc_seg, max_label_count = cc3d.connected_components((brain[bs][2] > 1), return_N=True)
        #     cc_seg[cc_seg != 1] = 0
        #     # corners_bone = find_external_cube_corners(cc_seg, flag='bone')[:4]
        #     corners_bone = get_end_points(cc_seg, flag='bone')[:4]
        #
        #     # print('number of labels', max_label_count)
        #     # print('max_label', np.max(cc_seg))
        #     # TODO 1. find the points matches the regression points. 2. create the distmap 3. create point corner dist map
        #
        #     # TODO 4. create attention layers to achieve the goal
        #     end_points = corners_big[:2]+corners_bone[:2]+corners_small[:2]+corners_bone[2:]+corners_small[2:]
        #     print('end_points', np.asarray(end_points))
        #     corner_heatmap = generate_gaussian_heatmap((1, 12, 128, 128, 128), np.asarray(end_points), sigma=5)
        #     corner_heatmap = corner_heatmap.cuda()
        # # print(corner_heatmap.shape)
        # # print(c8.shape)
        # c8f = torch.zeros((1, 12, 128, 128, 128)).cuda()
        # for i in range(12):
        #     heatmap_tensor = c8[:, i].unsqueeze(1)
        #     corner_tensor = corner_heatmap[:, i].unsqueeze(1)
        #     fused_tensor = torch.concat((heatmap_tensor, corner_tensor), dim=1)
        #     c8f[0, i] = self.fusion_layers[i](fused_tensor) + heatmap_tensor

        return c8, c8p
