r"""
    Preprocess DIP-IMU and TotalCapture dataset.
    Synthesize AMASS dataset.

    Please refer to the `paths` in `config.py` and set the path of each dataset correctly.
"""

import articulate as art
import torch
import os
import pickle
from config import paths, amass_data
import numpy as np
from tqdm import tqdm
import glob

smpl_mirror = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 23, 22]
ori_mirror = [1,0,3,2,4,5]
acc_mirror = [1,0,3,2,4,5]

def process_amass_split(smooth_n=4, mode='train'):
    train_split = ["BioMotionLab_NTroje", "BMLhandball", "BMLmovi", "CMU", "MPI_mosh", "DanceDB", "Eyes_Japan_Dataset", "MPI_HDM05", "KIT"]
    val_split = ["ACCAD", "DFaust67", "SFU", "EKUT", "HumanEva", "SSM_synced", "MPI_Limits"]
    used_datasets = train_split if mode == 'train' else val_split
    def _syn_acc(v):
        r"""
        Synthesize accelerations from vertex positions.
        """
        mid = smooth_n // 2
        acc = torch.stack([(v[i] + v[i + 2] - 2 * v[i + 1]) * 3600 for i in range(0, v.shape[0] - 2)])
        acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))
        if mid != 0:
            acc[smooth_n:-smooth_n] = torch.stack(
                [(v[i] + v[i + smooth_n * 2] - 2 * v[i + smooth_n]) * 3600 / smooth_n ** 2
                 for i in range(0, v.shape[0] - smooth_n * 2)])
        return acc

    vi_mask = torch.tensor([1961, 5424, 1177, 4662, 411, 3021]) #under knee
    ji_mask = torch.tensor([18, 19, 4, 5, 15, 0])
    body_model = art.ParametricModel(paths.smpl_file)

    data_pose, data_trans, data_beta, length = [], [], [], []
    id_list = []
    for ds_name in used_datasets:
        print('\rReading', ds_name)
        for npz_fname in tqdm(glob.glob(os.path.join(paths.raw_amass_dir, ds_name, ds_name, '*/*_poses.npz'))):
            try: cdata = np.load(npz_fname)
            except: continue

            framerate = int(cdata['mocap_framerate'])
            if framerate == 120: step = 2
            elif framerate == 60 or framerate == 59: step = 1
            else: continue

            id_list.append('_'.join(npz_fname.split('os.sep')[-3:]))
            data_pose.extend(cdata['poses'][::step].astype(np.float32))
            data_trans.extend(cdata['trans'][::step].astype(np.float32))
            data_beta.append(cdata['betas'][:10])
            length.append(cdata['poses'][::step].shape[0])

    assert len(data_pose) != 0, 'AMASS dataset not found. Check config.py or comment the function process_amass()'
    length = torch.tensor(length, dtype=torch.int)
    shape = torch.tensor(np.asarray(data_beta, np.float32))
    tran = torch.tensor(np.asarray(data_trans, np.float32))
    pose = torch.tensor(np.asarray(data_pose, np.float32)).view(-1, 52, 3)
    pose[:, 23] = pose[:, 37]     # right hand
    pose = pose[:, :24].clone()   # only use body

    # align AMASS global fame with DIP
    amass_rot = torch.tensor([[[1, 0, 0], [0, 0, 1], [0, -1, 0.]]])
    tran = amass_rot.matmul(tran.unsqueeze(-1)).view_as(tran)
    pose[:, 0] = art.math.rotation_matrix_to_axis_angle(
        amass_rot.matmul(art.math.axis_angle_to_rotation_matrix(pose[:, 0])))

    print('Synthesizing IMU accelerations and orientations')
    b = 0
    out_pose, out_shape, out_joint, out_vrot, out_vacc, out_pose_g, out_pose_g2 = [], [], [], [], [], [], []
    for i, l in tqdm(list(enumerate(length))):
        if l <= 12: b += l; print('\tdiscard one sequence with length', l); continue
        p = art.math.axis_angle_to_rotation_matrix(pose[b:b + l]).view(-1, 24, 3, 3)
        pg = body_model.forward_kinematics_R(p.clone())
        grot, joint, vert = body_model.forward_kinematics(p, shape[i], tran[b:b + l], calc_mesh=True)
        out_pose.append(pose[b:b + l].clone())  # N, 24, 3
        out_pose_g.append(grot.clone())  # N, 24, 3
        out_pose_g2.append(pg.clone())  # N, 24, 3
        out_shape.append(shape[i].clone())  # 10
        out_joint.append(joint[:, :24].contiguous().clone())  # N, 24, 3
        out_vacc.append(_syn_acc(vert[:, vi_mask]))  # N, 6, 3
        out_vrot.append(grot[:, ji_mask])  # N, 6, 3, 3
        b += l

    save_path = os.path.join(paths.amass_dir_pre, mode)
    print('Saving')
    os.makedirs(save_path, exist_ok=True)
    pickle.dump(id_list, open(os.path.join(save_path, 'id.pkl'), 'wb'))
    torch.save(out_pose, os.path.join(save_path, 'pose.pt'))
    torch.save(out_pose_g, os.path.join(save_path, 'pose_global.pt'))
    torch.save(out_pose_g2, os.path.join(save_path, 'pose_global_v2.pt'))
    torch.save(out_shape, os.path.join(save_path, 'shape.pt'))
    torch.save(out_joint, os.path.join(save_path, 'joint.pt'))
    torch.save(out_vrot, os.path.join(save_path, 'vrot.pt'))
    torch.save(out_vacc, os.path.join(save_path, 'vacc.pt'))
    print('Synthetic AMASS dataset is saved at', save_path)

def process_amass_split_sym(smooth_n=4, mode='train'):
    train_split = ["BioMotionLab_NTroje", "BMLhandball", "BMLmovi", "CMU", "MPI_mosh", "DanceDB", "Eyes_Japan_Dataset", "MPI_HDM05", "KIT"]
    val_split = ["ACCAD", "DFaust67", "SFU", "EKUT", "HumanEva", "SSM_synced", "MPI_Limits"]
    used_datasets = train_split if mode == 'train' else val_split
    def _syn_acc(v):
        r"""
        Synthesize accelerations from vertex positions.
        """
        mid = smooth_n // 2
        acc = torch.stack([(v[i] + v[i + 2] - 2 * v[i + 1]) * 3600 for i in range(0, v.shape[0] - 2)])
        acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))
        if mid != 0:
            acc[smooth_n:-smooth_n] = torch.stack(
                [(v[i] + v[i + smooth_n * 2] - 2 * v[i + smooth_n]) * 3600 / smooth_n ** 2
                 for i in range(0, v.shape[0] - smooth_n * 2)])
        return acc

    rot_mirror = torch.tensor([1.0, -1.0, -1.0])

    vi_mask = torch.tensor([1961, 5424, 1177, 4662, 411, 3021]) #under knee
    ji_mask = torch.tensor([18, 19, 4, 5, 15, 0])
    body_model = art.ParametricModel(paths.smpl_file)

    data_pose, data_trans, data_beta, length = [], [], [], []
    id_list = []
    for ds_name in used_datasets:
        print('\rReading', ds_name)
        for npz_fname in tqdm(glob.glob(os.path.join(paths.raw_amass_dir, ds_name, ds_name, '*/*_poses.npz'))):
            try: cdata = np.load(npz_fname)
            except: continue

            framerate = int(cdata['mocap_framerate'])
            if framerate == 120: step = 2
            elif framerate == 60 or framerate == 59: step = 1
            else: continue

            id_list.append('_'.join(npz_fname.split('os.sep')[-3:]))
            data_pose.extend(cdata['poses'][::step].astype(np.float32))
            data_trans.extend(cdata['trans'][::step].astype(np.float32))
            data_beta.append(cdata['betas'][:10])
            length.append(cdata['poses'][::step].shape[0])

    assert len(data_pose) != 0, 'AMASS dataset not found. Check config.py or comment the function process_amass()'
    length = torch.tensor(length, dtype=torch.int)
    shape = torch.tensor(np.asarray(data_beta, np.float32))
    tran = torch.tensor(np.asarray(data_trans, np.float32))
    pose = torch.tensor(np.asarray(data_pose, np.float32)).view(-1, 52, 3)
    pose[:, 23] = pose[:, 37]     # right hand
    pose = pose[:, :24].clone()   # only use body

    # align AMASS global fame with DIP
    amass_rot = torch.tensor([[[1, 0, 0], [0, 0, 1], [0, -1, 0.]]])
    tran = amass_rot.matmul(tran.unsqueeze(-1)).view_as(tran)
    pose[:, 0] = art.math.rotation_matrix_to_axis_angle(
        amass_rot.matmul(art.math.axis_angle_to_rotation_matrix(pose[:, 0])))

    print('Synthesizing IMU accelerations and orientations')
    b = 0
    out_pose, out_shape, out_joint, out_vrot, out_vacc, out_pose_g = [], [], [], [], [], []
    for i, l in tqdm(list(enumerate(length))):
        if l <= 12: b += l; print('\tdiscard one sequence with length', l); continue
        p = pose[b:b+l].view(-1, 24, 3)
        pm = p.clone()[:, smpl_mirror]
        pm = pm * rot_mirror
        posem = pm.clone()
        p = art.math.axis_angle_to_rotation_matrix(p).view(-1, 24, 3, 3)
        pm = art.math.axis_angle_to_rotation_matrix(pm).view(-1, 24, 3, 3)
        grot, joint, vert = body_model.forward_kinematics(p, shape[i], tran[b:b + l], calc_mesh=True)
        grotm, jointm, vertm = body_model.forward_kinematics(pm, shape[i], tran[b:b + l], calc_mesh=True)
        out_pose.append(pose[b:b + l].clone())  # N, 24, 3
        out_pose.append(posem.clone())  # N, 24, 3
        out_pose_g.append(grot.clone())  # N, 24, 3
        out_pose_g.append(grotm.clone())  # N, 24, 3
        out_shape.append(shape[i].clone())  # 10
        out_shape.append(shape[i].clone())  # 10
        out_joint.append(joint[:, :24].contiguous().clone())  # N, 24, 3
        out_joint.append(jointm[:, :24].contiguous().clone())  # N, 24, 3
        out_vacc.append(_syn_acc(vert[:, vi_mask]))  # N, 6, 3
        out_vacc.append(_syn_acc(vertm[:, vi_mask]))  # N, 6, 3
        out_vrot.append(grot[:, ji_mask])  # N, 6, 3, 3
        out_vrot.append(grotm[:, ji_mask])  # N, 6, 3, 3
        b += l

    save_path = os.path.join(paths.amass_dir_pre_sym, mode)
    print('Saving')
    os.makedirs(save_path, exist_ok=True)
    pickle.dump(id_list, open(os.path.join(save_path, 'id.pkl'), 'wb'))
    torch.save(out_pose, os.path.join(save_path, 'pose.pt'))
    torch.save(out_pose_g, os.path.join(save_path, 'pose_global.pt'))
    torch.save(out_shape, os.path.join(save_path, 'shape.pt'))
    torch.save(out_joint, os.path.join(save_path, 'joint.pt'))
    torch.save(out_vrot, os.path.join(save_path, 'vrot.pt'))
    torch.save(out_vacc, os.path.join(save_path, 'vacc.pt'))
    print('Synthetic AMASS dataset is saved at', save_path)

def process_dipimu():
    imu_mask = [7, 8, 11, 12, 0, 2]
    test_split = ['s_09', 's_10']
    accs, oris, poses = [], [], []
    ids = []

    for subject_name in test_split:
        for motion_name in os.listdir(os.path.join(paths.raw_dipimu_dir, subject_name)):
            path = os.path.join(paths.raw_dipimu_dir, subject_name, motion_name)
            data = pickle.load(open(path, 'rb'), encoding='latin1')
            acc = torch.from_numpy(data['imu_acc'][:, imu_mask]).float()
            ori = torch.from_numpy(data['imu_ori'][:, imu_mask]).float()
            pose = torch.from_numpy(data['gt']).float()
            # fill nan with nearest neighbors
            for _ in range(4):
                acc[1:].masked_scatter_(torch.isnan(acc[1:]), acc[:-1][torch.isnan(acc[1:])])
                ori[1:].masked_scatter_(torch.isnan(ori[1:]), ori[:-1][torch.isnan(ori[1:])])
                acc[:-1].masked_scatter_(torch.isnan(acc[:-1]), acc[1:][torch.isnan(acc[:-1])])
                ori[:-1].masked_scatter_(torch.isnan(ori[:-1]), ori[1:][torch.isnan(ori[:-1])])

            acc, ori, pose = acc[6:-6], ori[6:-6], pose[6:-6]
            if torch.isnan(acc).sum() == 0 and torch.isnan(ori).sum() == 0 and torch.isnan(pose).sum() == 0:
                accs.append(acc.clone())
                oris.append(ori.clone())
                poses.append(pose.clone())
                ids.append('_'.join(['DIP_IMU', subject_name, motion_name]))
            else:
                print('DIP-IMU: %s/%s has too much nan! Discard!' % (subject_name, motion_name))

    os.makedirs(paths.dipimu_dir_pre, exist_ok=True)
    torch.save({'acc': accs, 'ori': oris, 'pose': poses, 'id':ids}, os.path.join(paths.dipimu_dir_pre, 'test.pt'))
    print('Preprocessed DIP-IMU dataset is saved at', paths.dipimu_dir_pre)

def process_dipimu_train():
    imu_mask = [7, 8, 11, 12, 0, 2]
    test_split = ['s_09', 's_10']
    accs, oris, poses, out_joints, pose_global, pose_global2 = [], [], [], [], [], []
    ids = []
    valid_file_id_bases = ['s_01_05', 's_03_05', 's_07_04']
    body_model = art.ParametricModel(paths.smpl_file)

    for subject_name in sorted(os.listdir(paths.raw_dipimu_dir)):
        if subject_name not in test_split:
            print(subject_name)
            for motion_name in sorted(os.listdir(os.path.join(paths.raw_dipimu_dir, subject_name))):
                file_id_base = subject_name + '_' + motion_name[:-4]
                if file_id_base not in valid_file_id_bases:
                    path = os.path.join(paths.raw_dipimu_dir, subject_name, motion_name)
                    data = pickle.load(open(path, 'rb'), encoding='latin1')
                    acc = torch.from_numpy(data['imu_acc'][:, imu_mask]).float()
                    ori = torch.from_numpy(data['imu_ori'][:, imu_mask]).float()
                    pose = torch.from_numpy(data['gt']).float()

                    #get joint positions
                    p = art.math.axis_angle_to_rotation_matrix(pose).view(-1, 24, 3, 3)
                    pg2 = body_model.forward_kinematics_R(p.clone()) #uses zero shape and zero trans
                    pg, joint = body_model.forward_kinematics(p, calc_mesh=False) #uses zero shape and zero trans

                    # fill nan with nearest neighbors
                    for _ in range(4):
                        acc[1:].masked_scatter_(torch.isnan(acc[1:]), acc[:-1][torch.isnan(acc[1:])])
                        ori[1:].masked_scatter_(torch.isnan(ori[1:]), ori[:-1][torch.isnan(ori[1:])])
                        acc[:-1].masked_scatter_(torch.isnan(acc[:-1]), acc[1:][torch.isnan(acc[:-1])])
                        ori[:-1].masked_scatter_(torch.isnan(ori[:-1]), ori[1:][torch.isnan(ori[:-1])])

                    acc, ori, pose, pg, pg2 = acc[6:-6], ori[6:-6], pose[6:-6], pg[6:-6], pg2[6:-6]
                    joint = joint[6:-6]
                    if torch.isnan(acc).sum() == 0 and torch.isnan(ori).sum() == 0 and torch.isnan(pose).sum() == 0:
                        accs.append(acc.clone())
                        oris.append(ori.clone())
                        poses.append(pose.clone())
                        pose_global.append(pg.clone())
                        pose_global2.append(pg2.clone())
                        out_joints.append(joint[:, :24].contiguous().clone())  # N, 24, 3
                        ids.append('_'.join(['DIP_IMU', subject_name, motion_name]))
                    else:
                        print('DIP-IMU: %s/%s has too much nan! Discard!' % (subject_name, motion_name))

    os.makedirs(paths.dipimu_dir_pre, exist_ok=True)
    torch.save({'acc': accs, 'ori': oris, 'pose': poses, 'pose_global': pose_global, 'pose_global_v2': pose_global2, 'id':ids, 'full_pos':out_joints}, os.path.join(paths.dipimu_dir_pre, 'train.pt'))
    print('Preprocessed DIP-IMU dataset is saved at', paths.dipimu_dir_pre)
    
def process_dipimu_train_sym():
    rot_mirror = torch.tensor([1.0, -1.0, -1.0])
    tra_mirror = torch.tensor([-1.0, 1.0, 1.0])

    imu_mask = [7, 8, 11, 12, 0, 2]
    test_split = ['s_09', 's_10']
    accs, oris, poses, out_joints, pose_global = [], [], [], [], []
    ids = []
    valid_file_id_bases = ['s_01_05', 's_03_05', 's_07_04']
    body_model = art.ParametricModel(paths.smpl_file)

    for subject_name in sorted(os.listdir(paths.raw_dipimu_dir)):
        if subject_name not in test_split:
            print(subject_name)
            for motion_name in sorted(os.listdir(os.path.join(paths.raw_dipimu_dir, subject_name))):
                file_id_base = subject_name + '_' + motion_name[:-4]
                if file_id_base not in valid_file_id_bases:
                    path = os.path.join(paths.raw_dipimu_dir, subject_name, motion_name)
                    data = pickle.load(open(path, 'rb'), encoding='latin1')
                    acc = torch.from_numpy(data['imu_acc'][:, imu_mask]).float()
                    ori = torch.from_numpy(data['imu_ori'][:, imu_mask]).float()
                    pose = torch.from_numpy(data['gt']).float()

                    #get joint positions
                    pm = pose.clone().view(-1,24,3)[:, smpl_mirror]
                    pm = pm * rot_mirror
                    posem = pm.clone()
                    p = art.math.axis_angle_to_rotation_matrix(pose).view(-1, 24, 3, 3)
                    pm = art.math.axis_angle_to_rotation_matrix(pm).view(-1, 24, 3, 3)
                    pg, joint = body_model.forward_kinematics(p, calc_mesh=False) #uses zero shape and zero trans
                    pgm, jointm = body_model.forward_kinematics(pm, calc_mesh=False) #uses zero shape and zero trans

                    # fill nan with nearest neighbors
                    for _ in range(4):
                        acc[1:].masked_scatter_(torch.isnan(acc[1:]), acc[:-1][torch.isnan(acc[1:])])
                        ori[1:].masked_scatter_(torch.isnan(ori[1:]), ori[:-1][torch.isnan(ori[1:])])
                        acc[:-1].masked_scatter_(torch.isnan(acc[:-1]), acc[1:][torch.isnan(acc[:-1])])
                        ori[:-1].masked_scatter_(torch.isnan(ori[:-1]), ori[1:][torch.isnan(ori[:-1])])

                    acc_shape = acc.shape
                    ori_shape = ori.shape
                    
                    accm = acc.clone().view(-1, 6, 3)[:,acc_mirror]
                    accm = accm.view(-1, 3) * tra_mirror
                    accm = accm.view(acc_shape)

                    orim = ori.clone().view(-1, 6, 3, 3)[:,ori_mirror]
                    orim = art.math.rotation_matrix_to_axis_angle(orim)
                    orim = orim.view(-1, 3) * rot_mirror
                    orim = art.math.axis_angle_to_rotation_matrix(orim)
                    orim = orim.view(ori_shape)

                    acc, ori, pose, pg = acc[6:-6], ori[6:-6], pose[6:-6], pg[6:-6]
                    accm, orim, posem, pgm = accm[6:-6], orim[6:-6], posem[6:-6], pgm[6:-6]
                    joint, jointm = joint[6:-6], jointm[6:-6]
                    if torch.isnan(acc).sum() == 0 and torch.isnan(ori).sum() == 0 and torch.isnan(pose).sum() == 0:
                        accs.append(acc.clone())
                        accs.append(accm.clone())
                        oris.append(ori.clone())
                        oris.append(orim.clone())
                        poses.append(pose.clone())
                        poses.append(posem.clone())
                        pose_global.append(pg.clone())
                        pose_global.append(pgm.clone())
                        out_joints.append(joint[:, :24].contiguous().clone())  # N, 24, 3
                        out_joints.append(jointm[:, :24].contiguous().clone())  # N, 24, 3
                        ids.append('_'.join(['DIP_IMU', subject_name, motion_name]))
                        ids.append('_'.join(['DIP_IMU', subject_name, motion_name]))
                    else:
                        print('DIP-IMU: %s/%s has too much nan! Discard!' % (subject_name, motion_name))

    os.makedirs(paths.dipimu_dir_pre_sym, exist_ok=True)
    torch.save({'acc': accs, 'ori': oris, 'pose': poses, 'pose_global': pose_global, 'id':ids, 'full_pos':out_joints}, os.path.join(paths.dipimu_dir_pre_sym, 'train.pt'))
    print('Preprocessed DIP-IMU dataset is saved at', paths.dipimu_dir_pre_sym)

def process_dipimu_valid_sym():
    rot_mirror = torch.tensor([1.0, -1.0, -1.0])
    tra_mirror = torch.tensor([-1.0, 1.0, 1.0])

    imu_mask = [7, 8, 11, 12, 0, 2]
    test_split = ['s_09', 's_10']
    accs, oris, poses, out_joints, pose_global = [], [], [], [], []
    ids = []
    valid_file_id_bases = ['s_01_05', 's_03_05', 's_07_04']
    body_model = art.ParametricModel(paths.smpl_file)

    for subject_name in sorted(os.listdir(paths.raw_dipimu_dir)):
        if subject_name not in test_split:
            print(subject_name)
            for motion_name in sorted(os.listdir(os.path.join(paths.raw_dipimu_dir, subject_name))):
                file_id_base = subject_name + '_' + motion_name[:-4]
                if file_id_base in valid_file_id_bases:
                    path = os.path.join(paths.raw_dipimu_dir, subject_name, motion_name)
                    data = pickle.load(open(path, 'rb'), encoding='latin1')
                    acc = torch.from_numpy(data['imu_acc'][:, imu_mask]).float()
                    ori = torch.from_numpy(data['imu_ori'][:, imu_mask]).float()
                    pose = torch.from_numpy(data['gt']).float()

                    #get joint positions
                    pm = pose.clone().view(-1,24,3)[:, smpl_mirror]
                    pm = pm * rot_mirror
                    posem = pm.clone()
                    p = art.math.axis_angle_to_rotation_matrix(pose).view(-1, 24, 3, 3)
                    pm = art.math.axis_angle_to_rotation_matrix(pm).view(-1, 24, 3, 3)
                    pg, joint = body_model.forward_kinematics(p, calc_mesh=False) #uses zero shape and zero trans
                    pgm, jointm = body_model.forward_kinematics(pm, calc_mesh=False) #uses zero shape and zero trans

                    # fill nan with nearest neighbors
                    for _ in range(4):
                        acc[1:].masked_scatter_(torch.isnan(acc[1:]), acc[:-1][torch.isnan(acc[1:])])
                        ori[1:].masked_scatter_(torch.isnan(ori[1:]), ori[:-1][torch.isnan(ori[1:])])
                        acc[:-1].masked_scatter_(torch.isnan(acc[:-1]), acc[1:][torch.isnan(acc[:-1])])
                        ori[:-1].masked_scatter_(torch.isnan(ori[:-1]), ori[1:][torch.isnan(ori[:-1])])

                    acc_shape = acc.shape
                    ori_shape = ori.shape
                    
                    accm = acc.clone().view(-1, 6, 3)[:,acc_mirror]
                    accm = accm.view(-1, 3) * tra_mirror
                    accm = accm.view(acc_shape)

                    orim = ori.clone().view(-1, 6, 3, 3)[:,ori_mirror]
                    orim = art.math.rotation_matrix_to_axis_angle(orim)
                    orim = orim.view(-1, 3) * rot_mirror
                    orim = art.math.axis_angle_to_rotation_matrix(orim)
                    orim = orim.view(ori_shape)

                    acc, ori, pose, pg = acc[6:-6], ori[6:-6], pose[6:-6], pg[6:-6]
                    accm, orim, posem, pgm = accm[6:-6], orim[6:-6], posem[6:-6], pgm[6:-6]
                    joint, jointm = joint[6:-6], jointm[6:-6]
                    if torch.isnan(acc).sum() == 0 and torch.isnan(ori).sum() == 0 and torch.isnan(pose).sum() == 0:
                        accs.append(acc.clone())
                        accs.append(accm.clone())
                        oris.append(ori.clone())
                        oris.append(orim.clone())
                        poses.append(pose.clone())
                        poses.append(posem.clone())
                        pose_global.append(pg.clone())
                        pose_global.append(pgm.clone())
                        out_joints.append(joint[:, :24].contiguous().clone())  # N, 24, 3
                        out_joints.append(jointm[:, :24].contiguous().clone())  # N, 24, 3
                        ids.append('_'.join(['DIP_IMU', subject_name, motion_name]))
                        ids.append('_'.join(['DIP_IMU', subject_name, motion_name]))
                    else:
                        print('DIP-IMU: %s/%s has too much nan! Discard!' % (subject_name, motion_name))

    os.makedirs(paths.dipimu_dir_pre_sym, exist_ok=True)
    torch.save({'acc': accs, 'ori': oris, 'pose': poses, 'pose_global': pose_global, 'id':ids, 'full_pos':out_joints}, os.path.join(paths.dipimu_dir_pre_sym, 'valid.pt'))
    print('Preprocessed DIP-IMU dataset is saved at', paths.dipimu_dir_pre_sym)

def process_dipimu_valid():
    imu_mask = [7, 8, 11, 12, 0, 2]
    test_split = ['s_09', 's_10']
    accs, oris, poses, out_joints, pose_global, pose_global2 = [], [], [], [], [], []
    ids = []
    valid_file_id_bases = ['s_01_05', 's_03_05', 's_07_04']
    body_model = art.ParametricModel(paths.smpl_file)

    for subject_name in sorted(os.listdir(paths.raw_dipimu_dir)):
        if subject_name not in test_split:
            print(subject_name)
            for motion_name in sorted(os.listdir(os.path.join(paths.raw_dipimu_dir, subject_name))):
                file_id_base = subject_name + '_' + motion_name[:-4]
                if file_id_base in valid_file_id_bases:
                    path = os.path.join(paths.raw_dipimu_dir, subject_name, motion_name)
                    data = pickle.load(open(path, 'rb'), encoding='latin1')
                    acc = torch.from_numpy(data['imu_acc'][:, imu_mask]).float()
                    ori = torch.from_numpy(data['imu_ori'][:, imu_mask]).float()
                    pose = torch.from_numpy(data['gt']).float()

                    #get joint positions
                    p = art.math.axis_angle_to_rotation_matrix(pose).view(-1, 24, 3, 3)
                    pg2 = body_model.forward_kinematics_R(p.clone()).view(-1, 24, 3, 3)
                    pg, joint = body_model.forward_kinematics(p, calc_mesh=False) #uses zero shape and zero trans

                    # fill nan with nearest neighbors
                    for _ in range(4):
                        acc[1:].masked_scatter_(torch.isnan(acc[1:]), acc[:-1][torch.isnan(acc[1:])])
                        ori[1:].masked_scatter_(torch.isnan(ori[1:]), ori[:-1][torch.isnan(ori[1:])])
                        acc[:-1].masked_scatter_(torch.isnan(acc[:-1]), acc[1:][torch.isnan(acc[:-1])])
                        ori[:-1].masked_scatter_(torch.isnan(ori[:-1]), ori[1:][torch.isnan(ori[:-1])])

                    acc, ori, pose, pg, pg2 = acc[6:-6], ori[6:-6], pose[6:-6], pg[6:-6], pg2[6:-6]
                    joint = joint[6:-6]
                    if torch.isnan(acc).sum() == 0 and torch.isnan(ori).sum() == 0 and torch.isnan(pose).sum() == 0:
                        accs.append(acc.clone())
                        oris.append(ori.clone())
                        poses.append(pose.clone())
                        pose_global.append(pg.clone())
                        pose_global2.append(pg2.clone())
                        out_joints.append(joint[:, :24].contiguous().clone())  # N, 24, 3
                        ids.append('_'.join(['DIP_IMU', subject_name, motion_name]))
                    else:
                        print('DIP-IMU: %s/%s has too much nan! Discard!' % (subject_name, motion_name))

    os.makedirs(paths.dipimu_dir_pre, exist_ok=True)
    torch.save({'acc': accs, 'ori': oris, 'pose': poses, 'pose_global': pose_global, 'pose_global_v2': pose_global2, 'id':ids, 'full_pos':out_joints}, os.path.join(paths.dipimu_dir_pre, 'valid.pt'))
    print('Preprocessed DIP-IMU dataset is saved at', paths.dipimu_dir_pre)

def process_totalcapture():
    inches_to_meters = 0.0254
    file_name = 'gt_skel_gbl_pos.txt'
    body_model = art.ParametricModel(paths.smpl_file)

    accs, oris, poses, trans = [], [], [], []
    id_list = []
    out_joints = []
    #for file in sorted(os.listdir(paths.raw_totalcapture_dir)): #listdir causes weird problrms when loading pickled iles
    for file_ in sorted(glob.glob(os.path.join(paths.raw_totalcapture_dir, '*'))):
        file = os.path.basename(file_)
        data = pickle.load(open(os.path.join(paths.raw_totalcapture_dir, file), 'rb'), encoding='latin1')
        ori = torch.from_numpy(data['ori']).float()[:, torch.tensor([2, 3, 0, 1, 4, 5])]
        acc = torch.from_numpy(data['acc']).float()[:, torch.tensor([2, 3, 0, 1, 4, 5])]
        pose = torch.from_numpy(data['gt']).float().view(-1, 24, 3)

        # acc/ori and gt pose do not match in the dataset
        if acc.shape[0] < pose.shape[0]:
            pose = pose[:acc.shape[0]]
        elif acc.shape[0] > pose.shape[0]:
            acc = acc[:pose.shape[0]]
            ori = ori[:pose.shape[0]]

        assert acc.shape[0] == ori.shape[0] and ori.shape[0] == pose.shape[0]
        accs.append(acc)    # N, 6, 3
        oris.append(ori)    # N, 6, 3, 3
        poses.append(pose)  # N, 24, 3
        id_list.append(file)

        p = art.math.axis_angle_to_rotation_matrix(pose).view(-1, 24, 3, 3)
        _, joint = body_model.forward_kinematics(p, calc_mesh=False) #uses zero shape and zero trans
        out_joints.append(joint[:, :24].contiguous().clone())

    os.makedirs(paths.totalcapture_dir_pre, exist_ok=True)
    torch.save({'acc': accs, 'ori': oris, 'pose': poses, 'id':id_list, 'full_pos':out_joints},
               os.path.join(paths.totalcapture_dir_pre, 'test.pt'))
    print('Preprocessed TotalCapture dataset is saved at', paths.totalcapture_dir_pre)

if __name__ == '__main__':
    #process_dipimu()
    #process_totalcapture()
    #process_dipimu_train()
    #process_dipimu_valid()
    #process_dipimu_train_sym()
    #process_dipimu_valid_sym()
    #process_amass_split(mode='train')
    #process_amass_split_sym(mode='train')
    #process_amass_split(mode='valid')
    #process_amass_split_sym(mode='valid')
