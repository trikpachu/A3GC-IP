r"""
    Evaluate the pose estimation. args given should be model_path and hidden_units
"""

import torch
import tqdm
from net_aagc import PoseNetDIP
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
parser.add_argument('--path', help='The model to load', type=str, required=True)
parser.add_argument('--type', help='pretrain or finetuning', type=str, default='finetuning')
parser.add_argument('--gpu_index', type=int, default=0, help='gpu to use')
parser.add_argument('--norm', action='store_true', help='normalize the data ')
parser.add_argument('--rotsize', type=int, default=6, help='dimensions of the representation of the target rotation matrix (6 or 9)')
parser.add_argument('--hu', type=int, default=256, help='hidden units')
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
    if args.norm:
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
        inputs = torch.cat((acc[:,:-3], ori[:,:-9]), dim=-1)
        all_inputs.append(inputs.unsqueeze(0).to(device))
    return all_inputs

def evaluate_pose(dataset, num_past_frame=20, num_future_frame=5, model_path=''):
    #give as model_path directory containing the models, I will load the latest (before early_stopping) for each model. assuming the naming convention checkpoint_modelX_pretrain/finetuning_n.py
    #give also as additional argument pretrain or fine to discern which model to take.
    device = torch.device('cuda:{}'.format(args.gpu_index))
    model_path = args.path
    net = PoseNetDIP(rotsize=args.rotsize, n_hidden=args.hu, device=device).to(device)
    net.eval()
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['state_dict'])

    device = torch.device('cuda:0')
    evaluator = PoseEvaluator()

    print('loading data: ', os.path.join(dataset, 'test_tp.pt'))
    data = torch.load(os.path.join(dataset, 'test_tp.pt'))
    xs = prepare_input(data['ori'], data['acc'], device=device)
    ys = [reduced_to_full(p.view(-1, 15, 3, 3)).view(-1, 24, 3, 3).unsqueeze(1).to(device) for p in data['pose']]
    offline_errs = []
    import time
    offline_time = 0

    for x, y in tqdm.tqdm(list(zip(xs, ys))):
        s = time.time()
        pose_p_offline, _ = net.forward_offline(x)
        offline_time += time.time()-s

        pose_t = y
        offline_errs.append(evaluator.eval(pose_p_offline, pose_t))

    text_file_save_path = 'evaluation_results/{}.txt'.format(os.path.basename(args.path))
    os.makedirs('evaluation_results', exist_ok=True)
    txtfile = open(text_file_save_path, 'a')
    txtfile.write('Dataset: {}\n'.format(dataset))
    txtfile.write('offline time {}\n'.format(offline_time))
    print('============== offline time {} for {} ================'.format(offline_time, args.path))
    evaluator.print(torch.stack(offline_errs).mean(dim=0), txtfile)


if __name__ == '__main__':
    evaluate_pose(paths.dipimu_dir)
    evaluate_pose(paths.totalcapture_dir)


