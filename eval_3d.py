import os.path

import numpy as np
import pandas as pd
from scipy import ndimage
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import os
from dataloader import FetalDataset
from networks.apga import APGNet, APGANet
from networks.hourglass import StackedHourglass
from networks.network import UNeTR, ResUnet
from networks.nnmamba import nnMambaSeg
from networks.pseudo_resunet import ResUnetPseudo_3d, ResUnetPseudo
from networks.swinunetr_3d import SwinUNETR
from networks.vitpose import ViTPose3D
from evaluate_mre_vis import plot_3D_points, plot_3D_points_json, calculate_t_test
from networks.pseudo_resunet_gn import ResUnetPseudoP, ResUnetPseudoPD, ResUnetGN


def distance_points(p1, p2):
    return np.sqrt(((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2))


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


def get_points_from_mass_center(heatmap, number_points=6):
    final_points = np.zeros((1, number_points, 3))
    for i in range(number_points):
        heatmap_i = heatmap[i]
        point = ndimage.center_of_mass(heatmap_i)
        final_points[0, i] = point
    return final_points


def eval_model(model_name='resunet', checkpoint='./70-valid-4.7572test-4.5766.pth', refinement='', mode='test'):
    """
    :param model_name: name of model
    :param checkpoint: path of checkpoint
    :param refinement: 1. mass 2. align
    :return:
    """
    current_root = './results_'+mode+'/' + model_name + '/'
    if not os.path.exists(current_root):
        os.makedirs(current_root)
    test_set = FetalDataset(mode=mode, img_size=128, pseudo_label=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    if model_name == 'swin-unetr':
        model = SwinUNETR(img_size=(128, 128, 128), in_channels=1, out_channels=6).cuda()
    elif model_name == 'unetr':
        model = UNeTR(6).cuda()
    elif model_name == 'mtnet':
        model = ResUnet().cuda()
    elif model_name == 'resunet-bn':
        model = ResUnet().cuda()
    elif model_name == 'resunet-gn':
        model = ResUnetGN().cuda()
    elif model_name == 'vitpose':
        model = ViTPose3D().cuda()
    elif model_name == 'resunet3t':
        model = ResUnetPseudo_3d().cuda()
    elif model_name == 'pseudo_mt':
        model = ResUnetPseudo().cuda()
    elif model_name == 'pseudo_3t':
        model = ResUnetPseudoP().cuda()
    elif model_name == 'pseudo_3td':
        model = ResUnetPseudoPD().cuda()
    elif model_name == 'nnmamba':
        model = nnMambaSeg().cuda()
    elif model_name == 'apg':
        model = APGNet().cuda()
    elif model_name == 'apga':
        model = APGANet().cuda()
    elif model_name == 'hg':
        model = StackedHourglass().cuda()
    else:
        raise NotImplementedError

    state_dict = torch.load(checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    distance_losses = [[] for i in range(6)]
    data_idx = 0
    for data in tqdm(test_loader):
        data_idx += 1
        voxel, gt, case_info = data
        factor = case_info['factor']
        spacing = case_info['spacing'][0]
        name = case_info['name']

        origin = case_info['origin'].detach().cpu().numpy()[0]
        voxel = voxel.cuda().unsqueeze(1).float()

        if "point" in model_name:
            heatmap, out_coor, query_emb = model(voxel)
        elif 'apga' in model_name:
            heatmap, out_p, sup = model(voxel)
        elif "mt" in model_name or "apg" in model_name:
            heatmap, pseudo_seg = model(voxel)
        elif "mt" in model_name:
            heatmap, pseudo = model(voxel)
        elif "3td" in model_name:
            heatmap, out_p, out_pp, fg, att = model(voxel)
        elif "3t" in model_name:
            heatmap, _, _ = model(voxel)
        else:
            heatmap = model(voxel)

        if 'point' in model_name:
            pred_coor = np.asarray(out_coor['pred_points'].float().detach().cpu())
        elif 'mass' == refinement:
            pred_coor = get_points_from_heatmap_torch(heatmap).float().detach().cpu().numpy()
            # pred_coor = get_points_from_mass_center(np.asarray(heatmap.squeeze(0).float().detach().cpu()))
        elif 'prior' == refinement:
            refined_coors = pred_coor.copy()
            r1_2 = (pred_coor[0][0][1] + pred_coor[0][1][1]) / 2
            r2_2 = (pred_coor[0][2][0] + pred_coor[0][3][0] + pred_coor[0][4][0] + pred_coor[0][5][0]) / 4
            refined_coors[0][0][1] = refined_coors[0][1][1] = r1_2
            refined_coors[0][2][0] = refined_coors[0][3][0] = refined_coors[0][4][0] = refined_coors[0][5][0] = r2_2
            pred_coor = refined_coors
        else:
            pred_coor = get_points_from_heatmap_torch(heatmap).float().detach().cpu().numpy()

        gt = gt.float().numpy()
        norm_pred = pred_coor[0] / (factor[0].float() / spacing)
        os.makedirs(current_root+'/figs/', exist_ok=True)
        plot_3D_points(gt[0], pred_coor[0], save_name=current_root+'/figs/'+case_info['name'][0]+'.jpg')
        plot_3D_points_json(origin, norm_pred, case_info['json_path'], current_root)
        for b in range(voxel.shape[0]):
            pred_coor[b] /= (factor[b].float() / spacing)
            gt[b] /= (factor[b].float() / spacing)
            for idx in range(pred_coor.shape[1]):
                distance_losses[idx].append(distance_points(pred_coor[b][idx], gt[b][idx]))

    distance_array = np.round(np.array(distance_losses), decimals=3)

    df = pd.DataFrame(distance_array)
    df.to_csv(current_root + 'mre_each.csv', index=False)

    distance_category_mean = np.round(np.mean(distance_array, axis=1), decimals=3)
    distance_category_std = np.round(np.std(distance_array, axis=1), decimals=3)
    distance_all = np.round(np.mean(distance_category_mean), decimals=3).item()
    distance_all_std = np.round(np.std(distance_category_mean), decimals=3).item()

    print('dist_avg:', distance_category_mean)
    print('dist_std:', distance_category_std)
    print('dist_all:', distance_all)

    with open(current_root + 'mre_category.csv', 'w') as f:
        f.write(','.join(map(str, distance_category_mean)) + ',' + str(distance_all))

    distance_array = distance_array.flatten()
    with open(current_root + 'sdr.csv', 'w') as csv_file:
        for threshold in np.arange(0, 6.1, 0.1):
            rate = round(np.mean(distance_array <= threshold) * 100, 2)
            csv_file.write(f"{rate}, " if threshold < 6 else f"{rate}")
    print('finished: ' + model_name)
    return distance_category_mean.tolist()+[distance_all]


def format_results_as_latex(a, b, c):
    results = np.array([a, b, c])

    # Calculate mean and standard deviation across the first dimension (cross-validation folds)
    means = np.mean(results, axis=(0))
    stds = np.std(results, axis=(0))

    # Generate LaTeX formatted string
    latex_output = "& "
    latex_output += "& ".join([f"{mean:.2f}$_{{\\pm {std:.2f}}}$" for mean, std in zip(means, stds)])
    latex_output += "& \\textless{}0.0001 \\\\"
    print(latex_output)
    return latex_output


if __name__ == "__main__":
    # a = eval_model(model_name='resunet-bn', checkpoint='run/resunet-bn/f1-36-valid-1.4093-test-1.3887.pth')
    # b = eval_model(model_name='resunet-bn', checkpoint='run/resunet-bn/f2-44-valid-1.3833-test-1.3824.pth')
    # c = eval_model(model_name='resunet-bn', checkpoint='run/resunet-bn/f3-35-valid-1.4749-test-1.4123.pth')
    # format_results_as_latex(a, b, c)
    #
    # a = eval_model(model_name='hg', checkpoint='run/hg/f1-37-valid-1.4222-test-1.4044.pth')
    # b = eval_model(model_name='hg', checkpoint='run/hg/f2-46-valid-1.4217-test-1.392.pth')
    # c = eval_model(model_name='hg', checkpoint='run/hg/f3-42-valid-1.4357-test-1.4408.pth')
    # format_results_as_latex(a, b, c)
    #
    # a = eval_model(model_name='vitpose', checkpoint='run/vitpose/f1-47-valid-1.6155-test-1.646.pth')
    # b = eval_model(model_name='vitpose', checkpoint='run/vitpose/f2-57-valid-1.6138-test-1.6747.pth')
    # c = eval_model(model_name='vitpose', checkpoint='run/vitpose/f3-48-valid-1.6055-test-1.63.pth')
    # format_results_as_latex(a, b, c)
    #
    # a = eval_model(model_name='swin-unetr', checkpoint='run/swin-unetr/f1-43-valid-1.8114-test-1.7916.pth')
    # b = eval_model(model_name='swin-unetr', checkpoint='run/swin-unetr/f2-52-valid-1.7984-test-1.8138.pth')
    # c = eval_model(model_name='swin-unetr', checkpoint='run/swin-unetr/f3-59-valid-1.7824-test-1.7633.pth')
    # format_results_as_latex(a, b, c)

    # b = eval_model(model_name='apga', refinement='prior', checkpoint='run/apga/f2-47-valid-1.3641-test-1.3758.pth')
    # c = eval_model(model_name='apga', refinement='prior', checkpoint='run/apga/f3-51-valid-1.3603-test-1.37.pth')
    # a = eval_model(model_name='apga', refinement='prior', checkpoint='run/apga/f1-37-valid-1.3699-test-1.3652.pth')
    # format_results_as_latex(a, b, c)

    # a = eval_model(model_name='resunet-bn', checkpoint='run/resunet-bn/f1-36-valid-1.4093-test-1.3887.pth', mode='feta21')
    # b = eval_model(model_name='resunet-bn', checkpoint='run/resunet-bn/f2-44-valid-1.3833-test-1.3824.pth', mode='feta21')
    # c = eval_model(model_name='resunet-bn', checkpoint='run/resunet-bn/f3-35-valid-1.4749-test-1.4123.pth', mode='feta21')
    # format_results_as_latex(a, b, c)
    #
    # a = eval_model(model_name='hg', checkpoint='run/hg/f1-37-valid-1.4222-test-1.4044.pth', mode='feta21')
    # b = eval_model(model_name='hg', checkpoint='run/hg/f2-46-valid-1.4217-test-1.392.pth', mode='feta21')
    # c = eval_model(model_name='hg', checkpoint='run/hg/f3-42-valid-1.4357-test-1.4408.pth', mode='feta21')
    # format_results_as_latex(a, b, c)
    #
    # a = eval_model(model_name='vitpose', checkpoint='run/vitpose/f1-47-valid-1.6155-test-1.646.pth', mode='feta21')
    # b = eval_model(model_name='vitpose', checkpoint='run/vitpose/f2-57-valid-1.6138-test-1.6747.pth', mode='feta21')
    # c = eval_model(model_name='vitpose', checkpoint='run/vitpose/f3-48-valid-1.6055-test-1.63.pth', mode='feta21')
    # format_results_as_latex(a, b, c)
    #
    # a = eval_model(model_name='swin-unetr', checkpoint='run/swin-unetr/f1-43-valid-1.8114-test-1.7916.pth', mode='feta21')
    # b = eval_model(model_name='swin-unetr', checkpoint='run/swin-unetr/f2-52-valid-1.7984-test-1.8138.pth', mode='feta21')
    # c = eval_model(model_name='swin-unetr', checkpoint='run/swin-unetr/f3-59-valid-1.7824-test-1.7633.pth', mode='feta21')
    # format_results_as_latex(a, b, c)
    #
    # b = eval_model(model_name='apga', refinement='prior', checkpoint='run/apga/f2-47-valid-1.3641-test-1.3758.pth', mode='feta21')
    # c = eval_model(model_name='apga', refinement='prior', checkpoint='run/apga/f3-51-valid-1.3603-test-1.37.pth', mode='feta21')
    a = eval_model(model_name='apga', refinement='prior', checkpoint='run/apga/f1-37-valid-1.3699-test-1.3652.pth', mode='feta21')
    # format_results_as_latex(a, b, c)

    # t_stat, p_value = calculate_t_test(distance_array_b, distance_array_refine)  # 0.00011337965469972766
    # t_stat2, p_value2 = calculate_t_test(distance_array_1, distance_array_refine)  # 0.024925053209118482

    # a = eval_model(model_name='nnmamba', checkpoint='run/nnmamba/f1-43-valid-1.3448-test-1.3573.pth')
    # b = eval_model(model_name='nnmamba', checkpoint='run/nnmamba/f2-45-valid-1.3578-test-1.3336.pth')
    # c = eval_model(model_name='nnmamba', checkpoint='run/nnmamba/f3-29-valid-1.3602-test-1.3423.pth')
    # format_results_as_latex(a, b, c)
    #
    # a = eval_model(model_name='nnmamba', checkpoint='run/nnmamba/f1-43-valid-1.3448-test-1.3573.pth', mode='feta21')
    # b = eval_model(model_name='nnmamba', checkpoint='run/nnmamba/f2-45-valid-1.3578-test-1.3336.pth', mode='feta21')
    # c = eval_model(model_name='nnmamba', checkpoint='run/nnmamba/f3-29-valid-1.3602-test-1.3423.pth', mode='feta21')
    # format_results_as_latex(a, b, c)


