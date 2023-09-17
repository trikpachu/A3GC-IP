r"""
    Evaluate the pose estimation. args given should be model_path and hidden_units
"""

import torch
import tqdm
from net_aagc import PoseNet_GGRU
from config import *
import os
import articulate as art
import sys
import pickle 
import glob
import re
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--hu1', '--hidden_units1', default=256, type=int, help='hidden units')
parser.add_argument('--hu2', '--hidden_units2', default=256, type=int, help='hidden units')
parser.add_argument('--hu3', '--hidden_units3', default=256, type=int, help='hidden units')
parser.add_argument('--path', help='The model to load', type=str, required=True)
parser.add_argument('--mid1', type=int, default=-1, help='model index')
parser.add_argument('--mid2', type=int, default=-1, help='model index')
parser.add_argument('--mid3', type=int, default=-1, help='model index')
parser.add_argument('--type', help='pretrain or finetuning', type=str, default='finetuning')
parser.add_argument('--norm', action='store_true', help='normalize the data ')
parser.add_argument('--cda', action='store_true', help='use sym stats when norm True ')
parser.add_argument('--gpu_index', type=int, default=0, help='gpu to use')
args = parser.parse_args()

class PoseEvaluator:
    def __init__(self):
        self._eval_fn = art.FullMotionEvaluator(paths.male_smpl_file, joint_mask=torch.tensor([1, 2, 16, 17]))

    def eval(self, pose_p, pose_t):
        pose_p = pose_p.clone().view(-1, 24, 3, 3)
        pose_t = pose_t.clone().view(-1, 24, 3, 3)
        pose_p[:, joint_set.ignored] = torch.eye(3, device=pose_p.device)
        pose_t[:, joint_set.ignored] = torch.eye(3, device=pose_t.device)
        errs = self._eval_fn(pose_p, pose_t)
        return torch.stack([errs[9], errs[3], errs[0] * 100, errs[10] / 1000, errs[4] / 100])

    @staticmethod
    def print(errors, txtfile):
        for i, name in enumerate(['SIP Error (deg)', 'Angular Error (deg)', 'Positional Error (cm)',
                                  'jerk error (km/s^3)', 'Jitter Error (100m/s^3)']):
            print('%s: %.2f (+/- %.2f)' % (name, errors[i, 0], errors[i, 1]))
            txtfile.write('%s: %.2f (+/- %.2f)\n' % (name, errors[i, 0], errors[i, 1]))

def reduced_glb_to_full_local(glb_reduced_pose, m):
    global_full_pose = torch.eye(3, device=glb_reduced_pose.device).repeat(glb_reduced_pose.shape[0], 24, 1, 1)
    global_full_pose[:, joint_set.reduced] = glb_reduced_pose
    pose = m.inverse_kinematics_R(global_full_pose).view(-1, 24, 3, 3)
    pose[:, joint_set.ignored] = torch.eye(3, device=pose.device)
    return pose

def reduced_to_full(reduced_pose):
    full_pose = torch.eye(3, device=reduced_pose.device).repeat(reduced_pose.shape[0], 24, 1, 1)
    full_pose[:, joint_set.reduced] = reduced_pose
    return full_pose

def prepare_input(oris, accs, device):
    input_joints = [3, 4, 13, 14, 10]
    if args.norm:
        if args.cda:
            stats_data = torch.load('data/all_sym_train_stats.pt')
        else:
            stats_data = torch.load('data/all_train_stats.pt')
        ori_stats = stats_data['ori'] 
        acc_stats = stats_data['acc']
        ori_mean = ori_stats['mean_channel']
        ori_std = ori_stats['std_channel']
        acc_mean = acc_stats['mean_channel']
        acc_std = acc_stats['std_channel']
        def norm_input(ori, acc):
            ori_operated = (ori - ori_mean) / ori_std
            acc_operated = (acc - acc_mean) / acc_std
            return ori_operated, acc_operated

    all_inputs = []
    for idx in range(len(oris)):
        ori = oris[idx].float()
        acc = accs[idx].float()
        if args.norm:
            ori, acc = norm_input(ori, acc)

        inputs = torch.zeros((ori.shape[0], 15, 12))
        inputs_ = torch.cat((acc.view(-1, 6, 3)[:,:5], ori.view(-1, 6, 9)[:,:5]), dim=-1)
        for i, el in enumerate(input_joints):
            inputs[:,el] = inputs_[:,i]
        all_inputs.append(inputs.unsqueeze(0).to(device))
    return all_inputs

def evaluate_pose(dataset, num_past_frame=20, num_future_frame=5, model_path=''):
    #give as model_path directory containing the models, I will load the latest (before early_stopping) for each model. assuming the naming convention checkpoint_modelX_pretrain/finetuning_n.py
    #give also as additional argument pretrain or fine to discern which model to take.
    model_path = args.path
    model_list = glob.glob(os.path.join(model_path, '*'))
    model_list = [i for i in model_list if args.type in i]
    model1_file = [i for i in model_list if 'model1' in i]
    model2_file = [i for i in model_list if 'model2' in i]
    model3_file = [i for i in model_list if 'model3' in i]
    model1_file = list(map(list, zip(*[[i, int(re.findall('_\d+', os.path.basename(i))[0][1:])] for i in model1_file])))
    model2_file = list(map(list, zip(*[[i, int(re.findall('_\d+', os.path.basename(i))[0][1:])] for i in model2_file])))
    model3_file = list(map(list, zip(*[[i, int(re.findall('_\d+', os.path.basename(i))[0][1:])] for i in model3_file])))
    if args.mid1 >= 0:
        model1_file = model1_file[0][np.where(model1_file[1]==args.mid1)[0][0]]
    else:
        model1_file = model1_file[0][np.argmax(model1_file[1])]
    if args.mid2 >= 0:
        model3_file = model2_file[0][np.where(model2_file[1]==args.mid2)[0][0]]
    else:
        model2_file = model2_file[0][np.argmax(model2_file[1])]
    if args.mid3 >= 0:
        model3_file = model3_file[0][np.where(model3_file[1]==args.mid3)[0][0]]
    else:
        model3_file = model3_file[0][np.argmax(model3_file[1])]
    print('Loading the following models: ')
    print(model1_file)
    print(model2_file)
    print(model3_file)

    device = torch.device('cuda:0')
    evaluator = PoseEvaluator()

    with open('./nira_template_15_norm.pkl', 'rb') as f:
        nira = pickle.load(f) 
    nira = torch.from_numpy(nira).float()

    net1 = PoseNet_GGRU(input_size=12, rotsize=3, adjacency=nira, n_hidden=args.hu1, device=device).to(device)
    net2 = PoseNet_GGRU(input_size=15, rotsize=3, adjacency=nira, n_hidden=args.hu2, device=device).to(device)
    net3 = PoseNet_GGRU(input_size=15, rotsize=9, adjacency=nira, n_hidden=args.hu3, device=device).to(device)

    net1.eval()
    net2.eval()
    net3.eval()

    checkpoint1 = torch.load(model1_file)
    checkpoint2 = torch.load(model2_file)
    checkpoint3 = torch.load(model3_file)
    net1.load_state_dict(checkpoint1['state_dict'])
    net2.load_state_dict(checkpoint2['state_dict'])
    net3.load_state_dict(checkpoint3['state_dict'])

    print('loading data: ', os.path.join(dataset, 'test_tp.pt'))
    data = torch.load(os.path.join(dataset, 'test_tp.pt'))
    xs = prepare_input(data['ori'], data['acc'], device=device)
    ys = [reduced_to_full(p.view(-1, 15, 3, 3)).view(-1, 24, 3, 3).unsqueeze(1).to(device) for p in data['pose']]
    offline_errs = []
    import time
    offline_time = 0

    for x, y in tqdm.tqdm(list(zip(xs, ys))):
        net1.reset()
        s = time.time()
        leaf_pos, _ = net1.forward_offline(x)
        input1 = torch.cat((x, leaf_pos.view(1, leaf_pos.shape[1], 15, 3)), dim=-1)
        full_pos, _ = net2.forward_offline(input1)
        input2 = torch.cat((x, full_pos.view(1, full_pos.shape[1], 15, 3)), dim=-1)
        pose_p_offline, _ = net3.forward_offline(input2)
        offline_time += time.time()-s

        pose_t = y
        offline_errs.append(evaluator.eval(pose_p_offline, pose_t))

    text_file_save_path = 'evaluation_results/{}_{}.txt'.format(os.path.basename(args.path), os.path.basename(model1_file).split('.')[0], os.path.basename(model2_file).split('.')[0], os.path.basename(model3_file).split('.')[0])
    os.makedirs('evaluation_results', exist_ok=True)
    txtfile = open(text_file_save_path, 'a')
    txtfile.write('Dataset: {}\n'.format(dataset))
    txtfile.write('offline time {}\n'.format(offline_time))
    print('============== offline time {} for {}, {}, {} ================'.format(offline_time, model1_file, model2_file, model3_file))
    evaluator.print(torch.stack(offline_errs).mean(dim=0), txtfile)


if __name__ == '__main__':
    evaluate_pose(paths.dipimu_dir)
    evaluate_pose(paths.totalcapture_dir)


