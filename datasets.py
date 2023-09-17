from torch.utils.data import DataLoader, Dataset
import torch
import articulate
import articulate as art
from config import paths, joint_set
import config
import os

class GraphDataset_tp(Dataset):
    def __init__(self, filepath, no_norm=True, rotsize=9, norm=False, sym=False):
        super(GraphDataset_tp, self).__init__()
        data = torch.load(filepath)
        self.pose = data['pose']
        self.ori = data['ori']
        self.acc = data['acc']
        self.leaf_pos = data['leaf_pos']
        self.full_pos = data['full_pos']
        self.norm = norm
        self.rotsize = rotsize
        self.input_joints = [3, 4, 13, 14, 10]
        self.leaf_nodes = [4,5,15,18,19] 
        self.leaf_nodes_reduced = [3,4,10,13,14] 
        self.smpl_major_joints = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
        if norm:
            if sym:
                stats_data = torch.load('data/all_sym_train_stats.pt')
            else:
                stats_data = torch.load('data/all_train_stats.pt')
            ori_stats = stats_data['ori'] 
            acc_stats = stats_data['acc']
            self.ori_mean = ori_stats['mean_channel']
            self.ori_std = ori_stats['std_channel']
            self.acc_mean = acc_stats['mean_channel']
            self.acc_std = acc_stats['std_channel']

        if rotsize == 6:
            self.m = articulate.ParametricModel(paths.male_smpl_file)
            self.global_to_local_pose = self.m.inverse_kinematics_R

    def norm_input(self, ori, acc):
        ori_operated = (ori - self.ori_mean) / self.ori_std
        acc_operated = (acc - self.acc_mean) / self.acc_std
        return ori_operated, acc_operated

    def __getitem__(self, idx):
        smpl = self.pose[idx].float()
        if self.rotsize == 6:
            smpl = art.math.rotation_matrix_to_r6d(smpl).view(-1, 90)
        ori = self.ori[idx].float()
        acc = self.acc[idx].float()
        if self.norm:
            ori, acc = self.norm_input(ori, acc)
        full_pos = self.full_pos[idx].float()
        full_pos_input = full_pos.clone() +  torch.normal(0.00, 0.025, size=full_pos.shape)

        inputs = torch.zeros((ori.shape[0], 15, 12))
        inputs_ = torch.cat((acc.view(-1, 6, 3)[:,:5], ori.view(-1, 6, 9)[:,:5]), dim=-1)
        for i, el in enumerate(self.input_joints):
            inputs[:,el] = inputs_[:,i]

        leaf_pos = torch.zeros((len(ori), 15, 3))
        leaf_pos_input = torch.zeros((len(ori), 15, 3))
        for i, el in enumerate(self.leaf_nodes_reduced):
            leaf_pos[:,el] = full_pos[:,self.leaf_nodes[i]]
            leaf_pos_input[:,el] = full_pos_input[:,self.leaf_nodes[i]]
        full_pos = full_pos[:, self.smpl_major_joints]
        full_pos_input = full_pos_input[:, self.smpl_major_joints]
        leaf_pos = torch.reshape(leaf_pos, (-1, 45))
        leaf_pos_input = torch.reshape(leaf_pos_input, (-1, 15, 3))
        full_pos = torch.reshape(full_pos, (-1, 45))
        full_pos_input = torch.reshape(full_pos_input, (-1, 15, 3))
                
        return inputs, leaf_pos_input, full_pos_input, leaf_pos, full_pos, smpl

    def __len__(self):
        return len(self.ori)

class Dataset_tp(Dataset):
    def __init__(self, filepath, rotsize=9, norm=False, sym=False):
        super(Dataset_tp, self).__init__()
        data = torch.load(filepath)
        self.pose = data['pose']
        self.ori = data['ori']
        self.acc = data['acc']
        self.full_pos = data['full_pos']
        self.leaf_pos = data['leaf_pos']
        self.rotsize = rotsize
        self.smpl_major_joints = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
        if sym:
            stats_data = torch.load('data/all_sym_train_stats.pt')
        else:
                stats_data = torch.load('data/all_train_stats.pt')
        ori_stats = stats_data['ori'] 
        acc_stats = stats_data['acc']
        self.ori_mean = ori_stats['mean_channel']
        self.ori_std = ori_stats['std_channel']
        self.acc_mean = acc_stats['mean_channel']
        self.acc_std = acc_stats['std_channel']
        self.norm = norm

    def norm_input(self, ori, acc):
        ori_operated = (ori - self.ori_mean) / self.ori_std
        acc_operated = (acc - self.acc_mean) / self.acc_std
        return ori_operated, acc_operated

    def __getitem__(self, idx):
        smpl = self.pose[idx].float()
        if self.rotsize == 6:
            smpl = art.math.rotation_matrix_to_r6d(smpl).view(-1, 90)
        ori = self.ori[idx].float()
        acc = self.acc[idx].float()
        if self.norm:
            ori, acc = self.norm_input(ori, acc)
        full_pos = self.full_pos[idx][:,1:].float()
        leaf_pos = self.leaf_pos[idx].float()
        full_pos_input = full_pos.clone() +  torch.normal(0.00, 0.025, size=full_pos.shape)
        leaf_pos_input = leaf_pos.clone() +  torch.normal(0.00, 0.04, size=leaf_pos.shape)

        inputs = torch.cat((acc, ori), dim=-1)
        leaf_pos = torch.reshape(leaf_pos, (-1, 15))
        leaf_pos_input = torch.reshape(leaf_pos_input, (-1, 15))
        full_pos = torch.reshape(full_pos, (-1, 69))
        full_pos_input = torch.reshape(full_pos_input, (-1, 69))
                
        return inputs, leaf_pos_input, full_pos_input, leaf_pos, full_pos, smpl

    def __len__(self):
        return len(self.ori)

class Dataset_dip(Dataset):
    def __init__(self, filepath, no_root=True):
        super(Dataset_dip, self).__init__()
        data = torch.load(filepath)
        self.pose = data['pose']
        self.ori = data['ori']
        self.acc = data['acc']
        stats_data = torch.load('data/all_train_stats.pt')
        ori_stats = stats_data['ori'] 
        acc_stats = stats_data['acc']
        self.ori_mean = ori_stats['mean_channel']
        self.ori_std = ori_stats['std_channel']
        self.acc_mean = acc_stats['mean_channel']
        self.acc_std = acc_stats['std_channel']
        if no_root:
            self.ori_mean = self.ori_mean[:-9]
            self.ori_std = self.ori_std[:-9]
            self.acc_mean = self.acc_mean[:-3]
            self.acc_std = self.acc_std[:-3]
        self.no_root = no_root

    def norm_input(self, ori, acc):
        ori_operated = (ori - self.ori_mean) / self.ori_std
        acc_operated = (acc - self.acc_mean) / self.acc_std
        return ori_operated, acc_operated

    def __getitem__(self, idx):
        smpl = self.pose[idx].float()
        ori = self.ori[idx].float()
        acc = self.acc[idx].float()
        if self.no_root:
            ori = ori.view(ori.shape[0], -1, 9)[:,:-1].view(ori.shape[0], -1)
            acc = acc.view(acc.shape[0], -1, 3)[:,:-1].view(acc.shape[0], -1)

        ori, acc = self.norm_input(ori, acc)

        inputs = torch.cat((acc, ori), dim=-1)
                
        return inputs, smpl

    def __len__(self):
        return len(self.ori)

