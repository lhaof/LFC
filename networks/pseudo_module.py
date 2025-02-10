from scipy.ndimage import distance_transform_edt

real = [[23.8771, 59.0130, 55.1328],
        [114.1000, 59.0130, 51.8269],
        [18.9294, 59.0130, 55.1328],
        [118.7566, 59.0130, 51.5514],
        [88.1973, 82.2857, 31.4407],
        [47.4515, 82.2857, 33.6446],
        [57.8065, 109.5788, 60.9192],
        [57.8065, 17.1207, 49.3328],
        [66.0645, 84.3239, 27.5838],
        [66.0645, 80.1057, 44.3887],
        [66.0645, 89.7139, 34.1956],
        [66.0645, 78.6997, 34.7466]]

import numpy as np
from scipy.spatial import cKDTree

points_bone = np.array([
    [18.9294, 59.0130, 55.1328],
    [118.7566, 59.0130, 51.5514],
    [57.8065, 109.5788, 60.9192],
    [57.8065, 17.1207, 49.3328],
    [11111, 11111, 11111],
    [11111, 11111, 11111]
])

points_big = np.array([
    [23.8771, 59.0130, 55.1328],
    [114.1000, 59.0130, 51.8269],
    [57.8065, 109.5788, 60.9192],
    [57.8065, 17.1207, 49.3328],
    [11111, 11111, 11111],
    [11111, 11111, 11111]
])

points_small = np.array([
    [88.1973, 82.2857, 31.4407],
    [47.4515, 82.2857, 33.6446],
    [66.0645, 84.3239, 27.5838],
    [66.0645, 80.1057, 44.3887],
    [66.0645, 89.7139, 34.1956],
    [66.0645, 78.6997, 34.7466]
])

points_small_ori = np.array([
    [88.1973, 82.2857, 31.4407],
    [47.4515, 82.2857, 33.6446],
    [66.0645, 84.3239, 27.5838],
    [66.0645, 80.1057, 44.3887],
    [66.0645, 89.7139, 34.1956],
    [66.0645, 78.6997, 34.7466]
])


# hull = ConvexHull(points)
# cube_shape = (128, 128, 128)
#
# xx, yy, zz = np.meshgrid(np.arange(cube_shape[0]), np.arange(cube_shape[1]), np.arange(cube_shape[2]), indexing='ij')
# grid_points = np.vstack((xx.ravel(), yy.ravel(), zz.ravel())).T
#
# delaunay = Delaunay(hull.points[hull.vertices])
# mask = delaunay.find_simplex(grid_points) >= 0
# mask_3d = mask.reshape(cube_shape)

def find_external_cube_corners(mask, flag='small'):
    if flag == 'big':
        points = points_big
    elif flag == 'small':
        points = points_small
    elif flag == 'bone':
        points = points_bone
    non_zero_indices = np.argwhere(mask)
    min_coords = np.min(non_zero_indices, axis=0)
    max_coords = np.max(non_zero_indices, axis=0)

    corners = [
        (min_coords[0], min_coords[1], min_coords[2]),
        (max_coords[0], min_coords[1], min_coords[2]),
        (min_coords[0], max_coords[1], min_coords[2]),
        (max_coords[0], max_coords[1], min_coords[2]),
        (min_coords[0], min_coords[1], max_coords[2]),
        (max_coords[0], max_coords[1], max_coords[2]),
    ]

    # Build a k-d tree with the corner points
    tree = cKDTree(corners)

    # Find the nearest corner point for each input point
    ordered_corners = []
    for point in points:
        _, idx = tree.query(point)
        ordered_corners.append(corners[idx])
    # print("Ordered corners of the largest external cube:", ordered_corners)
    return ordered_corners


# corners = find_external_cube_corners(mask_3d)

def get_end_points(mask_3d, flag='small'):
    if flag == 'big':
        points = points_big
    elif flag == 'small':
        points = points_small
    elif flag == 'bone':
        points = points_bone

    # Extract the coordinates of the mask's voxels
    mask_coords = np.argwhere(mask_3d)

    # Calculate the mass center of the 3D mask
    mass_center = np.mean(mask_coords, axis=0)

    # Initialize the end points in each direction
    end_points = {
        'top': [0, 0, 0],
        'down': [0, 0, 0],
        'front': [0, 0, 0],
        'back': [0, 0, 0],
        'left': [0, 0, 0],
        'right': [0, 0, 0],
    }

    # Iterate through the coordinates of the mask's voxels
    for point in mask_coords:
        direction_vector = point - mass_center

        # Update the end points in each direction
        for direction in end_points.keys():
            if sum(end_points[direction]) == 0:
                end_points[direction] = point
            else:
                current_vector = end_points[direction] - mass_center

                if direction == 'top' and direction_vector[2] > current_vector[2]:
                    end_points[direction] = point
                elif direction == 'down' and direction_vector[2] < current_vector[2]:
                    end_points[direction] = point
                elif direction == 'front' and direction_vector[1] > current_vector[1]:
                    end_points[direction] = point
                elif direction == 'back' and direction_vector[1] < current_vector[1]:
                    end_points[direction] = point
                elif direction == 'left' and direction_vector[0] < current_vector[0]:
                    end_points[direction] = point
                elif direction == 'right' and direction_vector[0] > current_vector[0]:
                    end_points[direction] = point

    # print("Mass Center:", mass_center)
    # print("End Points:")
    corners = []
    for direction, point in end_points.items():
        # print(f"{direction.capitalize()}: {point}")
        corners.append(point)
    # print(corners)
    # assert len(corners) == 6
    # Build a k-d tree with the corner points
    tree = cKDTree(corners)

    # Find the nearest corner point for each input point
    ordered_corners = []
    for point in points:
        _, idx = tree.query(point)
        ordered_corners.append(corners[idx])
    # print("Ordered corners of the largest external cube:", ordered_corners)
    return ordered_corners


def get_dilated_brain(mask_3d_tensor, dilate_kernel_size=7):
    # Define the erosion and dilation kernels
    dilation_kernel = torch.ones(1, 1, dilate_kernel_size, dilate_kernel_size, dilate_kernel_size).cuda()

    # Perform erosion and dilation operations
    dilation_result = F.conv3d(mask_3d_tensor, dilation_kernel,
                               padding=(dilate_kernel_size - 1) // 2).cuda() > 0
    return dilation_result.float()

    mask_3d_tensor = mask_3d_tensor > 0
    # Compute the region between the expanded 3D mask and corrosive 3D mask
    difference = dilation_result ^ mask_3d_tensor

    # Convert the result back to a numpy array if needed
    brain = difference.float() * 255
    return brain


def dilates(brain):
    mask_3d_tensor = brain

    # Define the erosion and dilation kernels
    erosion_kernel_size = 5
    dilate_kernel_size = 5

    erosion_kernel = torch.ones(1, 1, erosion_kernel_size, erosion_kernel_size, erosion_kernel_size).cuda()
    dilation_kernel = torch.ones(1, 1, dilate_kernel_size, dilate_kernel_size, dilate_kernel_size).cuda()

    # Perform erosion and dilation operations
    erosion_result = F.conv3d(mask_3d_tensor, erosion_kernel,
                              padding=(erosion_kernel_size - 1) // 2).cuda() == erosion_kernel_size ** 3
    dilation_result = F.conv3d(mask_3d_tensor, dilation_kernel,
                               padding=(dilate_kernel_size - 1) // 2).cuda() > 0

    # Compute the region between the expanded 3D mask and corrosive 3D mask
    difference = dilation_result ^ erosion_result

    # Convert the result back to a numpy array if needed
    brain = difference.float() * 255
    return brain


def obtain_distmap(small_brain):
    dist_map = torch.tensor(distance_transform_edt(small_brain)) * small_brain.float()
    normalized_distmap = (dist_map - torch.min(dist_map)) / (torch.max(dist_map) - torch.min(dist_map))
    inverse_distmap = (1 - normalized_distmap / 2) * 255
    return inverse_distmap


def generate_prior_map(sigma=5.0, weight=128):
    numbers, x, y, z = 12, 128, 128, 128
    grid_x, grid_y, grid_z = torch.meshgrid(torch.arange(x), torch.arange(y), torch.arange(z))
    heatmap = torch.cat((grid_x.unsqueeze(0), grid_y.unsqueeze(0), grid_z.unsqueeze(0)), dim=0)
    return heatmap

# generate_prior_map()
import torch
from torch import nn
import torch.nn.functional as F


class SelfAttention3D(nn.Module):
    def __init__(self, in_channels, heads=8):
        super(SelfAttention3D, self).__init__()

        assert in_channels % heads == 0, "in_channels should be divisible by number of heads"

        self.in_channels = in_channels
        self.heads = heads
        self.scale = (in_channels // heads) ** -0.5

        self.pred_class = nn.Conv3d(in_channels, 1, kernel_size=1, stride=1, padding=0)
        # self.pred_class = nn.Sequential(
        #     nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
        #     nn.GroupNorm(int(in_channels/4), in_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(in_channels, 1, kernel_size=1, stride=1, padding=0),
        #     nn.GroupNorm(1, 1),
        #     nn.ReLU(inplace=True),
        # )
        self.pred_class_fg = nn.Conv3d(in_channels, 1, kernel_size=1, stride=1, padding=0)
        self.pred_class_pg = nn.Conv3d(in_channels, 1, kernel_size=1, stride=1, padding=0)

        self.query = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.key = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.value = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

        # self.query = nn.Sequential(
        #     nn.Conv3d(in_channels, in_channels * 4, kernel_size=1, stride=1, padding=0),
        #     nn.GroupNorm(int(in_channels/2), in_channels * 4),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(in_channels * 4, in_channels, kernel_size=1, stride=1, padding=0),
        #     nn.GroupNorm(int(in_channels/8), in_channels),
        #     nn.ReLU(inplace=True),
        # )
        # self.key = nn.Sequential(
        #     nn.Conv3d(in_channels, in_channels * 4, kernel_size=1, stride=1, padding=0),
        #     nn.GroupNorm(int(in_channels/2), in_channels * 4),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(in_channels * 4, in_channels, kernel_size=1, stride=1, padding=0),
        #     nn.GroupNorm(int(in_channels/8), in_channels),
        #     nn.ReLU(inplace=True),
        # )
        # self.value = nn.Sequential(
        #     nn.Conv3d(in_channels, in_channels * 4, kernel_size=1, stride=1, padding=0),
        #     nn.GroupNorm(int(in_channels/2), in_channels * 4),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(in_channels * 4, in_channels, kernel_size=1, stride=1, padding=0),
        #     nn.GroupNorm(int(in_channels/8), in_channels),
        #     nn.ReLU(inplace=True),
        # )

    def forward(self, x):
        bs, C, d, h, w = x.size()
        sx = self.pred_class(x)
        sxp = torch.sigmoid(sx)
        # sxp = torch.tanh(sx)
        # sxp = sx
        q_sup = self.query(x)
        k_sup = self.key(x)
        # pg_pred = self.pred_class_pg(k_sup)
        # sx = torch.cat((fg_pred, pg_pred), dim=1)

        queries = q_sup.view(bs, self.heads, C // self.heads, d, h, w)
        keys = k_sup.view(bs, self.heads, C // self.heads, d, h, w)
        values = self.value(x).view(bs, self.heads, C // self.heads, d, h, w)
        prob_fg = sxp[:, 0].view((bs, 1, 1, d, h, w))
        # prob_bg = sxp[:, 1].view((bs, 1, 1, d, h, w))
        keys = keys * prob_fg
        # queries = queries * prob_bg
        energy = torch.einsum('abcdef,abcdef->abcdef', keys, queries) * self.scale
        attention = F.softmax(energy, -1)

        out = torch.einsum('abcdef,abcdef->abcdef', attention, values)
        out = out.view(bs, C, d, h, w)
        # TODO
        # out = torch.sigmoid(q_sup/255) * out
        return out, sx, k_sup