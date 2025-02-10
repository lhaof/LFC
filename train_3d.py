import os.path
import random

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
from torchmetrics import Dice

from dataloader import FetalDataset
from networks.apga import APGANet, APGNet
from networks.hourglass import StackedHourglass
from networks.network import UNeTR, ResUnet, ResUnetMT
from networks.nnmamba import nnMambaSeg
from networks.pseudo_module import get_dilated_brain
from networks.pseudo_resunet import ResUnetPseudo, ResUnetPseudo_3d
from networks.swinunetr_3d import SwinUNETR
from networks.vitpose import ViTPose3D

dice_score = Dice().cuda()


def distance_points(p1, p2):
    return np.sqrt(((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2))


def generate_ordinal_heatmap(a_shape, points, sigma=5.0):
    batch_size, numbers, x, y, z = a_shape
    grid_x, grid_y, grid_z = torch.meshgrid(torch.arange(x), torch.arange(y), torch.arange(z), indexing='ij')
    heatmap = torch.zeros((batch_size, numbers, x, y, z))

    for b in range(batch_size):
        for n in range(numbers):
            point = points[b, n]
            distance = (grid_x - point[0]) ** 2 + (grid_y - point[1]) ** 2 + (grid_z - point[2]) ** 2
            for sigma in [3, 7, 12]:
                heatmap[b, n] += (torch.exp(-distance / (2 * sigma ** 2)) > 0).float()
    heatmap = 255 * (heatmap - torch.min(heatmap)) / (torch.max(heatmap) - torch.min(heatmap))

    return heatmap


def random_rotate_3d(image, heatmap):
    B, C, D, H, W = image.shape
    angles = np.radians(np.random.uniform(-10, 10, size=3))
    cos_vals, sin_vals = np.cos(angles), np.sin(angles)

    Rx = torch.tensor([[1, 0, 0, 0], [0, cos_vals[0], sin_vals[0], 0], [0, -sin_vals[0], cos_vals[0], 0], [0, 0, 0, 1]],
                      dtype=torch.float32, device=image.device)
    Ry = torch.tensor([[cos_vals[1], 0, -sin_vals[1], 0], [0, 1, 0, 0], [sin_vals[1], 0, cos_vals[1], 0], [0, 0, 0, 1]],
                      dtype=torch.float32, device=image.device)
    Rz = torch.tensor([[cos_vals[2], sin_vals[2], 0, 0], [-sin_vals[2], cos_vals[2], 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                      dtype=torch.float32, device=image.device)

    R = Rx @ Ry @ Rz

    R = R[:3, :4].unsqueeze(0).repeat(B, 1, 1)

    grid = F.affine_grid(R, size=image.size(), align_corners=False)
    rotated_image = F.grid_sample(image, grid, align_corners=False)
    rotated_heatmap = F.grid_sample(heatmap, grid, align_corners=False)

    return rotated_image, rotated_heatmap


def generate_gaussian_heatmap(a_shape, points, sigma=5.0):
    batch_size, numbers, x, y, z = a_shape
    grid_x, grid_y, grid_z = torch.meshgrid(torch.arange(x), torch.arange(y), torch.arange(z), indexing='ij')
    heatmap = torch.zeros((batch_size, numbers, x, y, z))

    for b in range(batch_size):
        for n in range(numbers):
            point = points[b, n]
            distance = (grid_x - point[0]) ** 2 + (grid_y - point[1]) ** 2 + (grid_z - point[2]) ** 2
            heatmap[b, n] = torch.exp(-distance / (2 * sigma ** 2))
    heatmap = 255 * (heatmap - torch.min(heatmap)) / (torch.max(heatmap) - torch.min(heatmap))

    return heatmap


def get_points_from_heatmap_torch(heatmap):
    # Flatten the last three dimensions and find the index of the maximum value
    max_indices = torch.argmax(heatmap.view(heatmap.size(0), heatmap.size(1), -1), dim=2)

    # Calculate the 3D indices from the flattened index
    z_indices = max_indices % heatmap.size(4)
    y_indices = (max_indices // heatmap.size(4)) % heatmap.size(3)
    x_indices = max_indices // (heatmap.size(3) * heatmap.size(4))

    # Stack the coordinates into a tensor
    final_points = torch.stack([x_indices, y_indices, z_indices], dim=2)

    return final_points


def get_points_from_ordinal_map(heatmap):
    final_points = np.zeros((1, 12, 3))
    for i in range(12):
        heatmap_i = heatmap[i]
        point = ndimage.measurements.center_of_mass(heatmap_i)
        final_points[0, i] = point
    return final_points


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def evaluate_model(model, model_name, valid_loader, align=False):
    distance_losses = [[] for i in range(6)]
    model.eval()
    data_idx = 0
    for data in valid_loader:
        data_idx += 1
        voxel, gt, case_info = data
        factor = case_info['factor']
        voxel = voxel.cuda().unsqueeze(1).float()
        if "3td" in model_name:
            heatmap, out_p, out_pp, fg, att = model(voxel)
        elif "3t" in model_name:
            heatmap, _, _ = model(voxel)
        elif 'apga' in model_name:
            heatmap, out_p, sup = model(voxel)
        elif "mt" in model_name or "apg" in model_name:
            heatmap, pseudo_seg = model(voxel)
        else:
            heatmap = model(voxel)

        if 'ordinal' in model_name:
            pred_coor = get_points_from_ordinal_map(np.asarray(heatmap.squeeze(0).float().detach().cpu()))
        else:
            pred_coor = get_points_from_heatmap_torch(heatmap).float().detach().cpu().numpy()

        if align:
            refined_pred_coor = pred_coor.copy()
            r1_2 = (pred_coor[0][0][1] + pred_coor[0][1][1]) / 2
            r2_2 = (pred_coor[0][2][0] + pred_coor[0][3][0] + pred_coor[0][4][0] + pred_coor[0][5][0]) / 4
            refined_pred_coor[0][0][1] = refined_pred_coor[0][1][1] = r1_2
            refined_pred_coor[0][2][0] = refined_pred_coor[0][3][0] = pred_coor[0][4][0] = pred_coor[0][5][0] = r2_2
            pred_coor = refined_pred_coor
        gt = gt.float()

        for b in range(voxel.shape[0]):
            pred_coor[b] /= (factor[b].float() / 0.6)
            gt[b] /= (factor[b].float() / 0.6)
            for idx in range(pred_coor.shape[1]):
                distance_losses[idx].append(distance_points(pred_coor[b][idx], gt[b][idx]))

    distance_array = np.array(distance_losses)
    distance_category = np.mean(distance_array, axis=1)
    distance_all = np.mean(distance_category).item()

    print('dist_c:', distance_category)
    print('dist_all:', distance_all)
    return round(distance_all, 4), distance_category


def adjust_learning_rate(optimizer, epoch, total_epochs=50, warmup_epochs=5, initial_lr=0.001, power=2.0, step_size=10):
    """Adjusts learning rate using polynomial decay with step changes every specified number of epochs after warmup
    period"""
    if epoch < warmup_epochs:
        # Linear warmup of learning rate
        lr = initial_lr * (epoch + 1) / warmup_epochs
    else:
        # Calculate the effective epoch for stepping every 'step_size' epochs
        effective_epoch = (epoch - warmup_epochs) // step_size * step_size + warmup_epochs
        # Polynomial decay of learning rate
        lr = initial_lr * (1 - (effective_epoch - warmup_epochs) / (total_epochs - warmup_epochs)) ** power

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(model_name='unetr'):
    setup_seed(1234)

    train_set = FetalDataset(mode='train')
    valid_set = FetalDataset(mode='valid')
    test_set = FetalDataset(mode='test')

    train_loader = DataLoader(train_set, batch_size=2, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    print('Data have been loaded.')

    if model_name == 'swin-unetr':
        model = SwinUNETR(img_size=(128, 128, 128), in_channels=1, out_channels=6).cuda()
    elif model_name == 'hg':
        model = StackedHourglass().cuda()
    elif model_name == 'unetr':
        model = UNeTR(6).cuda()
    elif model_name == 'mtnet':
        model = ResUnetMT().cuda()
    elif model_name == 'resunet-bn':
        model = ResUnet().cuda()
    elif model_name == 'vitpose':
        model = ViTPose3D().cuda()
    elif model_name == 'resunet_hover':
        model = ResUnetPseudo_3d().cuda()
    elif model_name == 'pseudo_mt':
        model = ResUnetPseudo().cuda()
    elif model_name == 'apg':
        model = APGNet().cuda()
    elif model_name == 'apga':
        model = APGANet().cuda()
    elif model_name == 'nnmamba':
        model = nnMambaSeg().cuda()
    else:
        raise NotImplementedError

    optimizer = torch.optim.Adam(model.parameters())

    local_best = 100
    for epoch in range(80):
        adjust_learning_rate(optimizer, epoch)
        print('epoch:', epoch)
        total_train_loss = 0
        model.train()
        for data in train_loader:
            voxel, gt, case_info = data
            voxel = voxel.cuda().unsqueeze(1)
            gt_heatmap = generate_gaussian_heatmap((voxel.shape[0], 6, 128, 128, 128), gt).cuda()

            # gt_heatmap, voxel = aug_mask_and_img(gt_heatmap, voxel)
            # gt_new = get_points_from_heatmap_torch(gt_heatmap).detach().cpu()
            # gt_heatmap = generate_gaussian_heatmap((voxel.shape[0], 6, 128, 128, 128), gt_new).cuda()

            if "3td" in model_name:
                out, out_p, fg1, fg2, fg3 = model(voxel)
            elif "3t" in model_name:
                out, out_p, out_pp = model(voxel)
            elif "hover" in model_name:
                out, out_p, out_po = model(voxel)
            elif 'apga' in model_name:
                out, out_p, sup = model(voxel)
            elif 'mt' in model_name or 'apg' in model_name:
                out, out_p = model(voxel)
            else:
                out = model(voxel)

            # TODO Points for debug
            # loss_paired = paired_loss(pred_points.detach().cpu(), gt.detach().cpu(), voxel.detach().cpu())
            # mae = mean_absolute_error(pred_points.flatten(), gt.flatten().cuda())

            if model_name == 'resunet_hover':
                small_brain = torch.tensor(case_info['small_brain']).cuda().unsqueeze(1)
                gt_heatmap_inside = torch.zeros(gt_heatmap.shape).cuda()
                for bs in range(gt_heatmap_inside.shape[0]):
                    for line_idx in range(gt_heatmap_inside.shape[1]):
                        gt_heatmap_inside[bs][line_idx] = gt_heatmap[bs][line_idx] * small_brain[bs].float()
                gt_heatmap_outside = gt_heatmap - gt_heatmap_inside
                gt_binary_heatmap = gt_heatmap.clone()
                gt_binary_heatmap[gt_heatmap > 128] = 255
                gt_binary_heatmap[gt_heatmap <= 128] = 0
                loss = mse_loss(out, gt_heatmap) + mse_loss(out_p, gt_binary_heatmap) + mse_loss(out_po,
                                                                                                 gt_heatmap_inside)

            elif 'apg' in model_name:
                small_brain = case_info['small_brain'].cuda().unsqueeze(1).float()
                dilated_brain = small_brain  # get_dilated_brain(small_brain)
                loss_pseudo = (1 - dice_score(out_p, dilated_brain.int()))
                loss = mse_loss(out, gt_heatmap) + loss_pseudo

            elif 'apga' == model_name:
                small_brain = case_info['small_brain'].cuda().unsqueeze(1).float()
                loss_att_l = 0
                loss_att_p = 0
                size = (64, 64, 64)
                for i in range(3):
                    pseudo_mask = F.interpolate(small_brain, size=size // (2 ^ i), mode='trilinear', align_corners=False).int()
                    pseudo_heat = F.interpolate(gt_heatmap, size=size // (2 ^ i), mode='trilinear', align_corners=False)
                    loss_att_p += (1 - dice_score(sup[i * 2 + 1], pseudo_mask))
                    loss_att_l += (mse_loss(sup[i * 2], pseudo_heat))

                loss_pseudo = (1 - dice_score(out_p, small_brain.int()))
                loss = mse_loss(out, gt_heatmap) + loss_pseudo + loss_att_l + loss_att_p
            else:
                loss = mse_loss(out, gt_heatmap)

            # print('loss', loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_train_loss += loss.item()

        valid_loss, valid_distances = evaluate_model(model, model_name, valid_loader)
        if local_best > valid_loss:
            local_best = valid_loss
            test_loss, test_distances = evaluate_model(model, model_name, test_loader)
            weights_path = 'run/' + model_name + '/' + str(epoch) + '-valid-' + str(valid_loss) + '-test-' + str(
                test_loss) + '.pth'
            os.makedirs('run/' + model_name + '/', exist_ok=True)
            torch.save(model.state_dict(), weights_path)
        print('mae loss:', valid_loss)

    return 'finish'

def eval_performance():
    from fvcore.nn import FlopCountAnalysis, parameter_count_table


    def calculate_model_metrics(model_name):
        # Create an instance of the model based on the provided model name
        if model_name == "SwinUNETR":
            model = SwinUNETR(img_size=(128, 128, 128), in_channels=1, out_channels=6).cuda()
        elif model_name == "StackedHourglass":
            model = StackedHourglass().cuda()
        elif model_name == "ResUnet":
            model = ResUnet().cuda()
        elif model_name == "APGANet":
            model = APGANet().cuda()
        elif model_name == "ViTPose3D":
            model = ViTPose3D().cuda()
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

        # Set the model to evaluation mode
        model.eval()

        # Create a sample input tensor
        input_tensor = torch.randn(1, 1, 128, 128, 128).cuda()

        # Calculate the FLOPs
        flops = FlopCountAnalysis(model, input_tensor)
        total_flops = flops.total()

        # Calculate the number of parameters
        param_table = parameter_count_table(model)
        total_params = sum(p.numel() for p in model.parameters())

        # Print the results
        print(f"Model: {model_name}")
        print(f"Total FLOPs: {total_flops / 1e9:.2f} GFLOPs")
        print(f"Total Parameters: {total_params / 1e6:.2f} MB")
        print("-----------------------------")

    # Calculate metrics for each model
    models = ["APGANet"]
    for model_name in models:
        calculate_model_metrics(model_name)


if __name__ == "__main__":
    train('nnmamba')
    # train('resunet-bn')
    # train('hg')
    # train('apg')
    # train('apga')
    # train('unetr')
    # train('swin-unetr')
    # train('vitpose')
