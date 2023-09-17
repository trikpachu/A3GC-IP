r"""
    Config for paths, joint set, and normalizing scales.
"""

class paths:
    raw_dipimu_dir = 'data/raw_datasets/DIP_IMU'   # raw DIP-IMU dataset path (raw_dipimu_dir/s_01/*.pkl)
    dipimu_dir_pre = 'data/preprocessed/dip-imu'      # output path for the preprocessed DIP-IMU dataset
    dipimu_dir_pre_sym = 'data/preprocessed/dip-imu_sym'      # output path for the preprocessed DIP-IMU dataset
    dipimu_dir = 'data/dip-imu'      # output path for the processed DIP-IMU dataset
    dipimu_dir_sym = 'data/dip-imu_sym'      # output path for the processed DIP-IMU dataset

    # DIP recalculates the SMPL poses for TotalCapture dataset. You should acquire the pose data from the DIP authors.
    raw_totalcapture_dir = 'data/raw_datasets/TotalCapture/DIP_recalculate'  # contain ground-truth SMPL pose (*.pkl)
    totalcapture_dir_pre = 'data/preprocessed/total_capture'          # output path for the preprocessed TotalCapture dataset
    totalcapture_dir = 'data/total_capture'          # output path for the processed TotalCapture dataset

    raw_amass_dir = 'data/raw_datasets/AMASS'   # raw DIP-IMU dataset path (raw_dipimu_dir/s_01/*.pkl)
    amass_dir_pre = 'data/preprocessed/amass'      # output path for the preprocessed AMASS dataset
    amass_dir_pre_sym = 'data/preprocessed/amass_sym'      # output path for the preprocessed AMASS dataset
    amass_dir = 'data/amass'      # output path for the processed AMASS dataset
    amass_dir_sym = 'data/amass_sym'      # output path for the processed AMASS dataset

    male_smpl_file = 'models/basicModel_m_lbs_10_207_0_v1.1.0.pkl'              # official SMPL model path
    female_smpl_file = 'models/basicModel_f_lbs_10_207_0_v1.1.0.pkl'              # official SMPL model path

class joint_set:
    leaf = [7, 8, 12, 20, 21]
    full = list(range(1, 24))
    reduced = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
    ignored = [0, 7, 8, 10, 11, 20, 21, 22, 23]

    lower_body = [0, 1, 2, 4, 5, 7, 8, 10, 11]
    lower_body_parent = [None, 0, 0, 1, 2, 3, 4, 5, 6]

    sensor = [18, 19, 4, 5, 15, 0, 1, 2, 9] 
    dip_imu = [7, 8, 11, 12, 0, 2, 9, 10, 1]
    VERTEX_IDS = [1962, 5431, 1096, 4583, 412, 3021, 949, 4434, 3506]
    SMPL_SENSOR = ['L_Elbow', 'R_Elbow', 'L_Knee', 'R_Knee', 'Head', 'Pelvis']

    n_leaf = len(leaf)
    n_full = len(full)
    n_reduced = len(reduced)
    n_ignored = len(ignored)
