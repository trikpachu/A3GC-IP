import numpy as np
import pickle
import os
import glob
from tqdm import tqdm
from random import randint
from pathlib import Path
import cv2
import sys
import scipy
import json
import quaternion
import torch
import gc
import collections
import articulate as art
from config import paths
import copy
from shutil import copyfile


def cut_validation(data_path, chunk_size=300)
    count = 0
    newdir = data_path+'_chunked'
    if not os.path.exists(newdir):
        os.makedirs(newdir)

    file_list = glob.glob(os.path.join(data_path, '[0-9]*'))
    try:
        copyfile(os.path.join(data_path, 'stats.p'), os.path.join(newdir, 'stats.p'))
    except:
        pass

    for file_ in tqdm(file_list):
        chunk_count = 0
        with open(file_, 'rb') as f:
            full_sample = pickle.load(f)
        data_id = full_sample[0]
        file_id = full_sample[1]
        sequential_data = full_sample[2:]
        data_len = len(sequential_data[0])
        for i in range(data_len//chunk_size):
            sample_data = [d[chunk_size*i:(i+1)*chunk_size] for d in sequential_data]
            current_file_id = file_id + '_part_{}'.format(chunk_count)
            chunk_count += 1
            sample = [data_id, current_file_id] + sample_data
            with open(os.path.join(newdir, f'{count}.p'), 'wb') as f2:
                pickle.dump(sample, f2)
            count += 1

#some functions
SMPL_MAJOR_JOINTS = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
leaf_nodes = [7, 8, 12, 20, 21]

def normalize(ori, acc, root=5):
    #normalizes the ori and acc data and resorts them in the last step
    ori = np.reshape(ori, (-1, 6, 3, 3))
    acc = np.reshape(acc, (-1, 6, 3))
    for i in range(len(ori)):
        root_ori = ori[i, root]
        root_acc = acc[i, root]
        for n in range(6):
            ori[i, n] = np.dot(root_ori.T, ori[i, n])
            acc[i, n] = np.dot(root_ori.T, acc[i, n] - root_acc)
    ori = ori.tolist()
    acc = acc.tolist()
    for i in range(len(ori)):
        ori[i].pop(root)
        acc[i].pop(root)
    ori = np.array(ori)
    acc = np.array(acc)
    return np.reshape(ori, (-1, 45)), np.reshape(acc, (-1, 15))

acc_scale = 30
def normalize_transpose(glb_ori, glb_acc):
    glb_acc = np.reshape(glb_acc, (-1, 6, 3))
    glb_ori = np.reshape(glb_ori, (-1, 6, 3, 3))
    acc = (np.concatenate((glb_acc[:, :5] - glb_acc[:, 5:], glb_acc[:, 5:]), axis=1) @ glb_ori[:, -1]) / acc_scale
    ori = np.concatenate((glb_ori[:, 5:].transpose(0,1,3,2) @ glb_ori[:, :5], glb_ori[:, 5:]), axis=1)
    return ori.reshape(-1, 54), acc.reshape(-1,18)

def normalize_pos(full_pos, leaf_pos):
    leaf_pos = np.reshape(leaf_pos, (-1, 5, 3))
    full_pos = np.reshape(full_pos, (-1, 5, 3))
    root_pos = full_pos[:,0]
    leaf_pos = leaf_pos - root_pos
    full_pos = full_pos - root_pos
    return leaf_pos, full_pos
    
with open('basicModel_m_lbs_10_207_0_v1.0.0.pkl', 'rb') as f:
    data = pickle.load(f, encoding='latin1')
parent = data['kintree_table'][0].tolist()
reduced = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]

def _forward_tree(x_local):
    r"""
    Multiply/Add matrices along the tree branches. x_local [N, J, *]. parent [J].
    """
    x_global = [x_local[0]]
    for i in range(1, len(parent)):
        x_global.append(np.matmul(x_global[parent[i]], x_local[i]))
    return np.array(x_global)

def forward_kinematics_R(R_local):
    r"""
    :math:`R_global = FK(R_local)`

    Forward kinematics that computes the global rotation of each joint from local rotations.

    Notes
    -----
    A joint's *local* rotation is expressed in its parent's frame.

    A joint's *global* rotation is expressed in the base (root's parent) frame.

    R_local[:, i], parent[i] should be the local rotation and parent joint id of
    joint i. parent[i] must be smaller than i for any i > 0.

    Args
    -----
    :param R_local: Joint local rotation tensor in shape [*] that can reshape to
                    [num_joint, 3, 3] (rotation matrices).
    :param parent: Parent joint id list in shape [num_joint]. Use -1 or None for base id (parent[0]).
    :return: Joint global rotation, in shape [num_joint, 3, 3].
    """
    R_global = _forward_tree(R_local)
    return R_global

def _reduced_local_to_reduced_global_mat(smpl_pose):
    smpl_pose = np.reshape(smpl_pose, (15,3,3))
    local_full_pose = np.array([np.eye(3) for i in range(24)])
    local_full_pose[reduced] = smpl_pose
    pose = forward_kinematics_R(local_full_pose)
    pose = pose[reduced]
    return np.reshape(pose, (135))

def _local_to_reduced_global_mat(smpl_pose):
    local_full_pose = np.reshape(smpl_pose, (24,3,3))
    pose = forward_kinematics_R(local_full_pose)
    pose = pose[reduced]
    return np.reshape(pose, (135))

def transform_to_torch_data(path_):
    save_path = path_+'.pt'
    acc = []
    ori = []
    pose = []
    full_pos = []
    for file in tqdm(glob.glob(os.path.join(path_, '[0-9]*'))):
        with open(file, 'rb') as f:
            data = pickle.load(f)
            ori.append(data[2])
            acc.append(data[3])
            pose.append(data[6])
    torch.save({'acc':torch.from_numpy(np.array(acc)), 'ori':torch.from_numpy(np.array(ori)), 'pose':torch.from_numpy(np.array(pose))}, save_path)
    
def transform_to_torch_data_irregular(path_, smpl_idx=6):
    save_path = path_+'.pt'
    acc = []
    ori = []
    pose = []
    full_pos = []
    for file in tqdm(glob.glob(os.path.join(path_, '[0-9]*'))):
        with open(file, 'rb') as f:
            data = pickle.load(f)
            ori.append(torch.from_numpy(data[2]).reshape(-1, 54).float())
            acc.append(torch.from_numpy(data[3]).reshape(-1, 18).float())
            pose.append(torch.from_numpy(data[smpl_idx]).float())
    torch.save({'acc':acc, 'ori':ori, 'pose':pose}, save_path)
    
def transform_to_torch_data_with_pos(path_):
    save_path = path_+'_pos.pt'
    acc = []
    ori = []
    pose = []
    full_pos = []
    leaf_pos = []
    for file in tqdm(glob.glob(os.path.join(path_, '[0-9]*'))):
        with open(file, 'rb') as f:
            data = pickle.load(f)
            ori.append(data[2])
            acc.append(data[3])
            leaf_pos.append(data[4])
            full_pos.append(data[5])
            pose.append(data[6])
    torch.save({'acc':torch.from_numpy(np.array(acc)), 'ori':torch.from_numpy(np.array(ori)), 'pose':torch.from_numpy(np.array(pose)), 'leaf_pos':torch.from_numpy(np.array(leaf_pos)), 'full_pos':torch.from_numpy(np.array(full_pos))}, save_path)
    
def transform_to_torch_data_irregular_with_pos(path_):
    save_path = path_+'_pos.pt'
    acc = []
    ori = []
    pose = []
    leaf_pos = []
    full_pos = []
    for file in tqdm(glob.glob(os.path.join(path_, '[0-9]*'))):
        with open(file, 'rb') as f:
            data = pickle.load(f)
            ori.append(torch.from_numpy(data[2]))
            acc.append(torch.from_numpy(data[3]))
            leaf_pos.append(torch.from_numpy(data[4]))
            full_pos.append(torch.from_numpy(data[5]))
            pose.append(torch.from_numpy(data[6]))
    torch.save({'acc':acc, 'ori':ori, 'pose':pose, 'leaf_pos':leaf_pos, 'full_pos':full_pos}, save_path)
  
body_model = art.ParametricModel(paths.smpl_file)

def transform_amass_split(mode='train', amass_path=None, amass_savepath=None):
    if amass_path is None:
        raise ValueError('please provide amass_path')
    if amass_savepath is None:
        raise ValueError('please provide amass_savepath')
    load_path = os.path.join(amass_path, mode)
    ori = torch.load(os.path.join(load_path, 'vrot.pt'))
    acc = torch.load(os.path.join(load_path, 'vacc.pt'))
    smpl = torch.load(os.path.join(load_path, 'pose.pt'))
    shape = torch.load(os.path.join(load_path, 'shape.pt'))
    smpl = [art.math.axis_angle_to_rotation_matrix(p).view(-1, 24, 3, 3) for p in smpl]
    full_pos = []
    print('getting correct pos')
    for i in tqdm(range(len(smpl))):
        p = smpl[i]
        s = shape[i]
        p_ = p.clone()
        p_[:,0] = torch.eye(3)
        _, joint = body_model.forward_kinematics(p_, s, calc_mesh=False)
        full_pos.append(joint[:, :24].contiguous().clone())

    ori = [i.cpu().detach().numpy() for i in ori]
    acc = [i.cpu().detach().numpy() for i in acc]
    smpl = [i.cpu().detach().numpy() for i in smpl]
    full_pos = [i.cpu().detach().numpy() for i in full_pos]

    for i in range(len(full_pos)):
        assert np.allclose(full_pos[i][:, 0], 0.0)

    ori = [np.reshape(i, (-1, 54)) for i in ori]
    acc = [np.reshape(i, (-1, 18)) for i in acc]
    smpl_temp = smpl
    for i in range(len(smpl_temp)):
        smpl_temp[i][:,0] = np.eye(3)
    smplg_re = [np.array([_local_to_reduced_global_mat(sp) for sp in sp_tempi]) for sp_tempi in smpl_temp]    
    smpl = [np.reshape(i[:,SMPL_MAJOR_JOINTS], (-1,135)) for i in smpl]
    leaf_pos = [i[:,leaf_nodes] for i in full_pos]

    print('normalizing ori and acc')
    oritp, acctp = [], []
    for i in tqdm(range(len(ori))):
        oritp_, acctp_ = normalize_transpose(copy.copy(ori[i]), copy.copy(acc[i]))
        oritp.append(oritp_)
        acctp.append(acctp_)

    print('saving the data')
    for i in tqdm(range(len(ori5))):
        savepath_tpg = os.path.join(amass_savepath, mode+'_tp_global', '{}.p'.format(i))
        Path(os.path.dirname(savepath_tpg)).mkdir(exist_ok=True, parents=True)
        sample_tpg = ['_', '_', oritp[i], acctp[i], leaf_pos[i], full_pos[i], smplg_re[i]]
        with open(savepath_tpg, 'wb') as f:
            pickle.dump(sample_tpg, f)

def transform_dip(mode='train', dip_path=None, dip_save_path=None):
    assert mode in ['train', 'valid']
    if dip_path is None:
        raise ValueError('please provide dip_path')
    if dip_save_path is None:
        raise ValueError('please provide dip_save_path')
    data = torch.load(os.path.join(dip_path, f'{mode}.pt'))
    acc = [i.cpu().detach().numpy() for i in data['acc']]
    ori = [i.cpu().detach().numpy() for i in data['ori']]
    smpl = [art.math.axis_angle_to_rotation_matrix(p).view(-1, 24, 3, 3) for p in data['pose']]
    full_pos = []
    print('getting correct pos')
    for i in tqdm(range(len(smpl))):
        p = smpl[i]
        p_ = p.clone()
        p_[:,0] = torch.eye(3)
        _, joint = body_model.forward_kinematics(p_, calc_mesh=False)
        full_pos.append(joint[:, :24].contiguous().clone())
    smpl = [i.cpu().detach().numpy() for i in smpl]
    full_pos = [i.cpu().detach().numpy() for i in full_pos]
    ori = [np.reshape(i, (-1, 54)) for i in ori]
    acc = [np.reshape(i, (-1, 18)) for i in acc]
    smpl_temp = smpl
    for i in tqdm(range(len(smpl_temp))):
        smpl_temp[i][:,0] = np.eye(3)
    smplg_re = [np.array([_local_to_reduced_global_mat(sp) for sp in sp_tempi]) for sp_tempi in smpl_temp]    
    smpl = [np.reshape(i[:,SMPL_MAJOR_JOINTS], (-1,135)) for i in smpl]
    leaf_pos = [i[:,leaf_nodes] for i in full_pos]
    oritp, acctp = [], []
    for i in tqdm(range(len(ori))):
        oritp_, acctp_ = normalize_transpose(copy.copy(ori[i]), copy.copy(acc[i]))
        oritp.append(oritp_)
        acctp.append(acctp_)
    for i in tqdm(range(len(ori))):
        savepath_tpg = os.path.join(dip_savepath, f'{mode}_tp_global', '{}.p'.format(i))
        Path(os.path.dirname(savepath_tpg)).mkdir(exist_ok=True, parents=True)
        sample_tpg = ['_', '_', oritp[i], acctp[i], leaf_pos[i], full_pos[i], smplg_re[i]]
        with open(savepath_tpg, 'wb') as f:
            pickle.dump(sample_tpg, f)
  
def transform_test(mode='dip'):
    assert mode in ['dip', 'tc']
    if mode == 'dip':
        data = torch.load(os.path.join(paths.dipimu_dir_pre, 'test.pt'))
        savepath_base_tp = os.path.join(paths.dipimu_dir, 'test_tp')
    else:
        data = torch.load(os.path.join(paths.totalcapture_dir_pre, f'test.pt'))
        savepath_base_tp = os.path.join(paths.totalcapture_dir, 'test_tp')        
    acc = [i.cpu().detach().numpy() for i in data['acc']]
    ori = [i.cpu().detach().numpy() for i in data['ori']]
    smpl = [art.math.axis_angle_to_rotation_matrix(p).view(-1, 24, 3, 3) for p in data['pose']]    
    smpl = [i.cpu().detach().numpy() for i in smpl]
    ori = [np.reshape(i, (-1, 54)) for i in ori]
    acc = [np.reshape(i, (-1, 18)) for i in acc]
    smpl = [np.reshape(i[:,SMPL_MAJOR_JOINTS], (-1,135)) for i in smpl]
    oritp, acctp = [], []
    for i in tqdm(range(len(ori))):
        oritp_, acctp_ = normalize_transpose(ori[i], acc[i])
        oritp.append(oritp_)
        acctp.append(acctp_)
    for i in tqdm(range(len(ori))):
        print('removing faulty total capture sequence')
        if mode == 'tc':
            if len(oritp[i]) == 3560:
                continue  
        savepath_tp = os.path.join(savepath_base_tp, '{}.p'.format(i))
        Path(os.path.dirname(savepath_tp)).mkdir(exist_ok=True, parents=True)
        sample_tp = ['_', '_', oritp[i], acctp[i], smpl[i]]
        with open(savepath_tp, 'wb') as f:
            pickle.dump(sample_tp, f)

def save_all_stats(data_paths, test=False, sym=True):
    if sym:
        symornosym = '_sym'
    else:
        symornosym = ''
    all_data = {}
    if test:
        key_list = ['acc', 'ori', 'pose']
    else:
        key_list = ['acc', 'ori', 'pose', 'leaf_pos', 'full_pos']
    for key in key_list:
        all_data[key] = []
    for data_path in data_paths:
        data = torch.load(data_path)
        for key in key_list:
            if isinstance(data[key], list):
                cd = torch.cat(data[key])
            else:
                cd = torch.flatten(data[key], 0, 1)
            all_data[key].append(cd.clone())
        del data
    for key in key_list:
        for i in all_data[key]:
            print(i.shape)
        all_data[key] = torch.cat(all_data[key], dim=0)
        
    stats = {}
    for key1 in key_list:
        current_data = all_data[key1]
        stats[key1] = {}
        try:
            current_data = torch.cat(current_data, dim=0)
        except:
            print('shape before flatten: ', current_data.shape)
            #current_data = torch.flatten(current_data, 0, 1)
            #print('shape after flatten: ', current_data.shape)
        stats[key1]['std_channel'] = torch.std(current_data, dim=0)
        stats[key1]['mean_channel'] = torch.mean(current_data, dim=0)
        stats[key1]['std_all'] = torch.std(current_data)
        stats[key1]['mean_all'] = torch.mean(current_data)
    if test:
        savename = f'data/all{symornosym}_test_stats.pt'
    else:
        savename = f'data/all{symornosym}_train_stats.pt'
    torch.save(stats, savename)
    print('saved stats for ', savename)
    
if __name__ == '__main__':
    #Transform the synthetic AMASS data
    transform_amass_split(mode='train', amass_path=paths.amass_dir_pre_sym, amass_savepath=paths.amass_dir_sym)
    transform_amass_split(mode='valid', amass_path=paths.amass_dir_pre_sym, amass_savepath=paths.amass_dir_sym)
    cut_validation('data/amass_sym/train_tp_global')
    cut_validation('data/amass_sym/valid_tp_global')
    transform_to_torch_data_with_pos('data/amass_sym/train_tp_global_chunked')
    transform_to_torch_data_with_pos('data/amass_sym/valid_tp_global_chunked')
    #transform_amass_split(mode='train', amass_path=paths.amass_dir_pre, amass_savepath=paths.amass_dir)
    #transform_amass_split(mode='valid', amass_path=paths.amass_dir_pre, amass_savepath=paths.amass_dir)
    #cut_validation('data/amass/train_tp_global') 
    #cut_validation('data/amass/valid_tp_global')
    #transform_to_torch_data_with_pos('data/amass/train_tp_global_chunked')
    #transform_to_torch_data_with_pos('data/amass/valid_tp_global_chunked')

    #Transform the DIP-IMU train and validation data
    transform_dip('train', dip_path=paths.dipimu_dir_pre_sym, dip_save_path=dipimu_dir_sym)
    transform_dip('valid', dip_path=paths.dipimu_dir_pre_sym, dip_save_path=dipimu_dir_sym)
    cut_validation('data/dip-imu_sym/train_tp_global')
    cut_validation('data/dip-imu_sym/valid_tp_global')
    transform_to_torch_data_with_pos('data/dip-imu_sym/train_tp_global_chunked')
    transform_to_torch_data_with_pos('data/dip-imu_sym/valid_tp_global_chunked')
    #transform_dip('train', dip_path=paths.dipimu_dir_pre, dip_save_path=dipimu_dir)
    #transform_dip('valid', dip_path=paths.dipimu_dir_pre, dip_save_path=dipimu_dir)
    #cut_validation('data/dip-imu/train_tp_global')
    #cut_validation('data/dip-imu/valid_tp_global')
    #transform_to_torch_data_with_pos('data/dip-imu/train_tp_global_chunked')
    #transform_to_torch_data_with_pos('data/dip-imu/valid_tp_global_chunked')

    #Transform the test sets
    transform_test('dip')
    transform_test('tc')
    transform_to_torch_data_irregular('data/total_capture/test_tp', smpl_idx=4)
    transform_to_torch_data_irregular('data/dip-imu/test_tp', smpl_idx=4)

    #stats files are provided, but can be generated with this, note that the test data might be used for stats, as we only used the stats for the input, that is always available for test data          
    #save_all_stats([dip_train_path, dip_valid_path, amass_train_path, amass_valid_path], test=False)
    #save_all_stats([dip_train_path, dip_valid_path, amass_train_path, amass_valid_path, dip_test_path, total_capture_test_path], test=True)
