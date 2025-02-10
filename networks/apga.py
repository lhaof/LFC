import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import GroupNorm


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding."""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_type='BN'):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        if norm_type == 'BN':
            self.bn1 = nn.BatchNorm3d(planes)
            self.bn2 = nn.BatchNorm3d(planes)
        else:
            self.bn1 = nn.GroupNorm(4, planes)
            self.bn2 = nn.GroupNorm(4, planes)

        self.conv1 = conv3x3(inplanes, planes, stride)
        # self.bn1 = nn.BatchNorm3d(planes)

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        # self.bn2 = nn.BatchNorm3d(planes)
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


def make_res_layer(inplanes, planes, blocks, stride=1, norm_type='BN'):
    if norm_type == 'BN':
        downsample = nn.Sequential(
            conv1x1(inplanes, planes, stride),
            nn.BatchNorm3d(planes),
        )
    else:
        downsample = nn.Sequential(
            conv1x1(inplanes, planes, stride),
            nn.GroupNorm(4, planes),
        )

    layers = []
    layers.append(BasicBlock(inplanes, planes, stride, downsample, norm_type=norm_type))
    for _ in range(1, blocks):
        layers.append(BasicBlock(planes, planes, norm_type=norm_type))

    return nn.Sequential(*layers)


class DoubleConv(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3, norm_type='BN'):
        super(DoubleConv, self).__init__()
        if norm_type == 'BN':
            self.conv = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=int(kernel_size / 2)),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_ch, out_ch, 3, padding=1, dilation=1),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=int(kernel_size / 2)),
                nn.GroupNorm(4, out_ch),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_ch, out_ch, 3, padding=1, dilation=1),
                nn.GroupNorm(4, out_ch),
                nn.ReLU(inplace=True),
            )

    def forward(self, input):
        return self.conv(input)


class SingleConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(SingleConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv3d(in_ch, out_ch, 3, padding=1), GroupNorm(4, out_ch), nn.ReLU(inplace=True))

    def forward(self, input):
        return self.conv(input)


class APGNet(nn.Module):
    def __init__(self, in_ch=1, channels=32, blocks=3, out_ch=6, out_pseudo=1, norm_type='GN'):
        super(APGNet, self).__init__()

        self.in_conv = DoubleConv(in_ch, channels, stride=2, kernel_size=3, norm_type=norm_type)
        self.layer1 = make_res_layer(channels, channels * 2, blocks, stride=2, norm_type=norm_type)
        self.layer2 = make_res_layer(channels * 2, channels * 4, blocks, stride=2, norm_type=norm_type)
        self.layer3 = make_res_layer(channels * 4, channels * 8, blocks, stride=2, norm_type=norm_type)

        self.up5 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv5 = DoubleConv(channels * 12, channels * 4)
        self.up6 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv6 = DoubleConv(channels * 6, channels * 2)
        self.up7 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv7 = DoubleConv(channels * 3, channels)
        self.up8 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv8 = nn.Sequential(
            nn.Conv3d(channels, out_ch, kernel_size=3, stride=1, padding=int(3 / 2)),
            nn.GroupNorm(1, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, dilation=1),
            nn.GroupNorm(1, out_ch),
            nn.ReLU(inplace=True),
        )

        self.up5p = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv5p = DoubleConv(channels * 12, channels * 4)
        self.up6p = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv6p = DoubleConv(channels * 6, channels * 2)
        self.up7p = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv7p = DoubleConv(channels * 3, channels)
        self.up8p = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv8p = nn.Sequential(
            nn.Conv3d(channels, out_pseudo, kernel_size=3, stride=1, padding=int(3 / 2)),
            nn.GroupNorm(1, out_pseudo),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_pseudo, out_pseudo, 3, padding=1, dilation=1),
            nn.GroupNorm(1, out_pseudo),
            nn.ReLU(inplace=True),
        )

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

        return c8, c8p


class AttentionlayerFC(nn.Module):
    def __init__(self, dim, r=16):
        super(AttentionlayerFC, self).__init__()
        self.layer1 = nn.Linear(int(dim), int(dim // r))
        self.layer2 = nn.Linear(int(dim // r), dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inp):
        x = self.relu(self.layer1(inp))
        att = self.sigmoid(self.layer2(x))
        return att.unsqueeze(-1)

class Attentionlayer(nn.Module):
    def __init__(self, dim, r=16):
        super(Attentionlayer, self).__init__()
        self.layer1 = nn.Conv3d(int(dim // 2), int(dim // r), 1, 1, 0)
        self.layer2 = nn.Conv3d(int(dim // r), dim, 1, 1, 0)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inp):
        x = self.relu(self.layer1(inp))
        att = self.sigmoid(self.layer2(x))
        return att.unsqueeze(-1)


def max_pooling(input_tensor):
    # Apply max pooling across the spatial dimensions (d, h, w)
    max_pooled = F.max_pool3d(input_tensor, kernel_size=input_tensor.size()[2:])
    # Flatten the result
    return max_pooled.view(input_tensor.size(0), -1, 1)


def avg_pooling(input_tensor):
    # Apply average pooling across the spatial dimensions (d, h, w)
    avg_pooled = F.avg_pool3d(input_tensor, kernel_size=input_tensor.size()[2:])
    # Flatten the result
    return avg_pooled.view(input_tensor.size(0), -1, 1)


def min_pooling(input_tensor):
    # In PyTorch, there is no direct function for min pooling, so we invert the input to use max pooling
    inverted_input = -input_tensor
    # Apply max pooling on the inverted input
    min_pooled = -F.max_pool3d(inverted_input, kernel_size=inverted_input.size()[2:])
    # Flatten the result
    return min_pooled.view(input_tensor.size(0), -1, 1)


def stochastic_pooling(input_tensor):
    # Reshape the tensor to have all spatial elements in separate rows for each channel
    bs, C, d, h, w = input_tensor.shape
    flat_tensor = input_tensor.view(bs, C, -1)  # Now shape is (bs, C, d*h*w)

    # Compute probabilities for multinomial, treating each element in C as separate distribution
    probs = F.softmax(flat_tensor, dim=2)

    # Sample one index for each distribution
    indices = torch.multinomial(probs.view(-1, d*h*w), 1)  # Reshape to 2D for multinomial

    # Reshape indices to have the same batch and channel dimensions
    indices = indices.view(bs, C, 1)  # Now shape is (bs, C, 1)

    # Batch and channel indices to gather values from flat_tensor
    batch_indices = torch.arange(bs).view(-1, 1, 1).expand(bs, C, 1).to(input_tensor.device)
    channel_indices = torch.arange(C).view(1, -1, 1).expand(bs, C, 1).to(input_tensor.device)

    # Gather the values at the sampled indices
    gathered = flat_tensor[batch_indices, channel_indices, indices]

    # No need to flatten the result since we want to keep the channel dimension
    return gathered.reshape(bs, C, 1)  # This will be of shape (bs, C, 1)

class ScalingSelfAtt3Dot(nn.Module):
    def __init__(self, in_channels, heads=16):
        super(ScalingSelfAtt3Dot, self).__init__()

        assert in_channels % heads == 0, "in_channels should be divisible by number of heads"

        self.in_channels = in_channels
        self.heads = heads
        self.scale = (in_channels // heads) ** -0.5

        self.query = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.key = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.value = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

        self.query1_prior = nn.Conv3d(int(in_channels/2), 6, kernel_size=1, stride=1, padding=0)
        self.query2_prior = nn.Conv3d(int(in_channels/2), 1, kernel_size=1, stride=1, padding=0)

        self.squeeze1 = nn.Conv1d(3, 1, kernel_size=1, stride=1, padding=0)
        self.squeeze2 = nn.Conv1d(3, 1, kernel_size=1, stride=1, padding=0)
        self.se1 = Attentionlayer(in_channels, r=in_channels//16)
        self.se2 = Attentionlayer(in_channels, r=in_channels//16)
        self.sec1 = AttentionlayerFC(in_channels, r=in_channels//16)
        self.sec2 = AttentionlayerFC(in_channels, r=in_channels//16)

    def pooling(self, input_tensor):
        # Calculate pooling
        max_pooled_output = max_pooling(input_tensor)
        avg_pooled_output = avg_pooling(input_tensor)
        min_pooled_output = min_pooling(input_tensor)
        # stochastic_pooled_output = stochastic_pooling(input_tensor)
        return torch.cat((max_pooled_output, avg_pooled_output, min_pooled_output), dim=2)

    def forward(self, x):
        bs, C, d, h, w = x.size()

        q_sup = self.query(x)
        k_sup = self.key(x)
        v_sup = self.value(x)

        queries = q_sup.view(bs, self.heads, C // self.heads, d, h, w)
        keys = k_sup.view(bs, self.heads, C // self.heads, d, h, w)
        values = v_sup.view(bs, self.heads, C // self.heads, d, h, w)

        # Split the heads into two halves for queries, keys, and values
        query1, query2 = torch.split(queries, self.heads // 2, dim=1)
        key1, key2 = torch.split(keys, self.heads // 2, dim=1)
        value1, value2 = torch.split(values, self.heads // 2, dim=1)

        sup_q1 = self.query1_prior(query1.reshape(bs, int(C//2), d, h, w))
        sup_q2 = self.query2_prior(query2.reshape(bs, int(C//2), d, h, w))

        # Calculate the energy and apply attention for each half separately
        energy1 = torch.einsum('bncdhw,bncdhw->bncdhw', query1, key1) * self.scale
        attention1 = F.softmax(energy1, dim=2)
        out1 = torch.einsum('bncdhw,bncdhw->bncdhw', attention1, value1).reshape(bs, int(C//2), d, h, w)

        energy2 = torch.einsum('bncdhw,bncdhw->bncdhw', query2, key2) * self.scale
        attention2 = F.softmax(energy2, dim=2)
        out2 = torch.einsum('bncdhw,bncdhw->bncdhw', attention2, value2).reshape(bs, int(C//2), d, h, w)

        # queries = q_sup.view(bs, self.heads, C // self.heads, d, h, w)
        # keys = k_sup.view(bs, self.heads, C // self.heads, d, h, w)
        # values = v_sup.view(bs, self.heads, C // self.heads, d, h, w)
        #
        # energy = torch.einsum('abcdef,abcdef->abcdef', keys, queries) * self.scale
        # attention = F.softmax(energy, -1)
        #
        # out = torch.einsum('abcdef,abcdef->abcdef', attention, values)
        # out = out.view(bs, C, d, h, w)
        #
        # out1 = out[:, :C // 2, :, :, :]
        # out2 = out[:, C // 2:, :, :, :]

        # out1 = self.pooling(out1)
        # out2 = self.pooling(out2)
        # out1 = self.squeeze1(out1.transpose(1, 2)).squeeze(1)
        # out2 = self.squeeze2(out2.transpose(1, 2)).squeeze(1)

        out1_s = self.se1(out1).squeeze(-1)
        out2_s = self.se2(out2).squeeze(-1)

        out1 = self.pooling(out1_s)
        out2 = self.pooling(out2_s)
        out1 = self.squeeze1(out1.transpose(1, 2)).squeeze(1)
        out2 = self.squeeze2(out2.transpose(1, 2)).squeeze(1)

        out1 = self.sec1(out1).reshape(bs, C, 1, 1, 1) * out1_s
        out2 = self.sec2(out2).reshape(bs, C, 1, 1, 1) * out2_s

        return out1, out2, sup_q1, sup_q2


class ScalingSelfAtt3D(nn.Module):
    def __init__(self, in_channels, heads=8):
        super(ScalingSelfAtt3D, self).__init__()

        assert in_channels % heads == 0, "in_channels should be divisible by number of heads"

        self.in_channels = in_channels
        self.heads = heads
        self.scale = (in_channels // heads) ** -0.5

        self.query = nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.key = nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.value = nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

        self.squeeze1 = nn.Conv1d(3, 1, kernel_size=1, stride=1, padding=0)
        self.squeeze2 = nn.Conv1d(3, 1, kernel_size=1, stride=1, padding=0)
        self.se1 = Attentionlayer(in_channels, r=in_channels//16)
        self.se2 = Attentionlayer(in_channels, r=in_channels//16)

    def pooling(self, input_tensor):
        # Calculate pooling
        max_pooled_output = max_pooling(input_tensor)
        avg_pooled_output = avg_pooling(input_tensor)
        min_pooled_output = min_pooling(input_tensor)
        # stochastic_pooled_output = stochastic_pooling(input_tensor)
        return torch.cat((max_pooled_output, avg_pooled_output, min_pooled_output), dim=2)

    def forward(self, x):
        bs, C, d, h, w = x.size()
        x = self.pooling(x)
        x = self.squeeze1(x.transpose(1, 2)).transpose(1, 2)

        q_sup = self.query(x)
        k_sup = self.key(x)
        v_sup = self.value(x)

        queries = q_sup.view(bs, self.heads, C // self.heads)
        keys = k_sup.view(bs, self.heads, C // self.heads)
        values = v_sup.view(bs, self.heads, C // self.heads)

        # Split the heads into two halves for queries, keys, and values
        query1, query2 = torch.split(queries, self.heads // 2, dim=1)
        key1, key2 = torch.split(keys, self.heads // 2, dim=1)
        value1, value2 = torch.split(values, self.heads // 2, dim=1)

        # Calculate the energy and apply attention for each half separately
        energy1 = torch.einsum('bnc,bnc->bnc', query1, key1) * self.scale
        attention1 = F.softmax(energy1, dim=2)
        out1 = torch.einsum('bnc,bnc->bnc', attention1, value1).reshape(bs, int(C//2))

        energy2 = torch.einsum('bnc,bnc->bnc', query2, key2) * self.scale
        attention2 = F.softmax(energy2, dim=2)
        out2 = torch.einsum('bnc,bnc->bnc', attention2, value2).reshape(bs, int(C//2))

        out1 = self.se1(out1).reshape(bs, C, 1, 1, 1)
        out2 = self.se2(out2).reshape(bs, C, 1, 1, 1)

        return out1, out2, attention1, attention2


class APGANet(nn.Module):
    def __init__(self, in_ch=1, channels=32, blocks=3, out_ch=6, out_pseudo=1, norm_type='BN'):
        super(APGANet, self).__init__()

        self.in_conv = DoubleConv(in_ch, channels, stride=2, kernel_size=3, norm_type=norm_type)
        self.layer1 = make_res_layer(channels, channels * 2, blocks, stride=2, norm_type=norm_type)
        self.layer2 = make_res_layer(channels * 2, channels * 4, blocks, stride=2, norm_type=norm_type)
        self.layer3 = make_res_layer(channels * 4, channels * 8, blocks, stride=2, norm_type=norm_type)

        self.down1 = nn.Upsample(size=(16, 16, 16), mode='trilinear', align_corners=False)
        self.up1 = nn.Upsample(size=(64, 64, 64), mode='trilinear', align_corners=False)

        self.down2 = nn.Upsample(size=(16, 16, 16), mode='trilinear', align_corners=False)
        self.up2 = nn.Upsample(size=(32, 32, 32), mode='trilinear', align_corners=False)

        self.up5 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv5 = DoubleConv(channels * 12, channels * 4, norm_type=norm_type)
        self.up6 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv6 = DoubleConv(channels * 6, channels * 2, norm_type=norm_type)
        self.up7 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv7 = DoubleConv(channels * 3, channels, norm_type=norm_type)
        self.up8 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        if norm_type == 'BN':
            self.conv8 = DoubleConv(channels, out_ch, norm_type=norm_type)
        else:
            self.conv8 = nn.Sequential(
                nn.Conv3d(channels, out_ch, kernel_size=3, stride=1, padding=int(3 / 2)),
                nn.GroupNorm(1, out_ch),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_ch, out_ch, 3, padding=1, dilation=1),
                nn.GroupNorm(1, out_ch),
                nn.ReLU(inplace=True),
            )

        self.conv5p = DoubleConv(channels * 12, channels * 4, norm_type=norm_type)
        self.up6p = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv6p = DoubleConv(channels * 6, channels * 2, norm_type=norm_type)
        self.up7p = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv7p = DoubleConv(channels * 3, channels, norm_type=norm_type)
        self.up8p = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        if norm_type == 'BN':
            self.conv8p = DoubleConv(channels, out_pseudo, norm_type=norm_type)
        else:
            self.conv8p = nn.Sequential(
                nn.Conv3d(channels, out_pseudo, kernel_size=3, stride=1, padding=int(3 / 2)),
                nn.GroupNorm(1, out_pseudo),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_pseudo, out_pseudo, 3, padding=1, dilation=1),
                nn.GroupNorm(1, out_pseudo),
                nn.ReLU(inplace=True),
            )

        # attention:
        # self.decouple_att = TransformerDown_SPrune(in_channels=32, out_channels=32, image_size=(16, 16, 16))
        # self.decouple_att2 = TransformerDown_SPrune(in_channels=64, out_channels=64, image_size=(16, 16, 16))
        self.prior_att1 = ScalingSelfAtt3Dot(channels, heads=16)
        self.prior_att2 = ScalingSelfAtt3Dot(channels * 2, heads=16)
        self.prior_att3 = ScalingSelfAtt3Dot(channels * 4, heads=16)

    def forward(self, input):
        c1 = self.in_conv(input)  # bs, 32, 64, 64, 64
        scale_1, scale_1_p, sup_l1, sup_p1 = self.prior_att1(c1)

        c2 = self.layer1(c1)  # bs, 32, 32, 32, 32
        scale_2, scale_2_p, sup_l2, sup_p2 = self.prior_att2(c2)

        c3 = self.layer2(c2)  # bs, 32, 16, 16, 16
        scale_3, scale_3_p, sup_l3, sup_p3 = self.prior_att3(c3)

        c4 = self.layer3(c3)

        up_5 = self.up5(c4)

        merge5 = torch.cat([up_5, c3 * scale_3], dim=1)
        c5 = self.conv5(merge5)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c2 * scale_2], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c1 * scale_1], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        c8 = self.conv8(up_8)

        # skip unet for prediction
        merge5p = torch.cat([up_5, c3 * scale_3_p], dim=1)
        c5p = self.conv5p(merge5p)
        up_6p = self.up6p(c5p)
        merge6p = torch.cat([up_6p, c2 * scale_2_p], dim=1)
        c6p = self.conv6p(merge6p)
        up_7p = self.up7p(c6p)
        merge7p = torch.cat([up_7p, c1 * scale_1_p], dim=1)
        c7p = self.conv7p(merge7p)
        up_8p = self.up8p(c7p)
        c8p = self.conv8p(up_8p)

        return c8, c8p, (sup_l1, sup_p1, sup_l2, sup_p2, sup_l3, sup_p3)
