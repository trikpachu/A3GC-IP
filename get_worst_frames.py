'''
This script returns the 10 best/worst frames for the SOTA, our model and the comparison of the two.
If you want to use it, you need to change gt_path, gt2_path, tp_path and our_path. 
'''
import numpy as np
import quaternion
import cv2
import os
import glob
import pickle
import torch
import tqdm
from net_aagc import PoseNet3
from config import *
import os
import articulate.evaluator as art
from utils import normalize_and_concat, only_concat
import sys
import pickle 
import glob
import re
import numpy as np
import argparse
import copy

class PoseEvaluator:
    def __init__(self):
        self._eval_fn = art.FullMotionEvaluatorAngleList(paths.male_smpl_file, joint_mask=torch.tensor([1, 2, 16, 17]))

    def eval(self, pose_t, pose_p):
        pose_p = pose_p.clone().view(-1, 24, 3, 3)
        pose_t = pose_t.clone().view(-1, 24, 3, 3)
        pose_p[:, joint_set.ignored] = torch.eye(3, device=pose_p.device)
        pose_t[:, joint_set.ignored] = torch.eye(3, device=pose_t.device)
        errs = self._eval_fn(pose_p, pose_t)
        return errs.cpu().detach().numpy()

evaluator = PoseEvaluator()

datasets = ['tc', 'dip']
flat_dict = {'tp':[], 'gt':[], 'tp_err':[], 'our':[], 'our_err':[]}
our_angle_error_per_sequence = {'tc':[], 'dip':[]}
tp_angle_error_per_sequence = {'tc':[], 'dip':[]}
len_per_sequence = {'tc':[], 'dip':[]}
for ds in datasets:
    gt_path = f'evaluation_results/sequences/{ds}/m1/gt'
    gt2_path = f'evaluation_results/sequences/{ds}/m2/gt'
    tp_path = f'evaluation_results/sequences/{ds}/m1/pred'
    our_path = f'evaluation_results/sequences/{ds}/m2/pred'

    for sample in glob.glob(os.path.join(gt_path, '*')):
        sb = os.path.basename(sample)
        gt_data = pickle.load(open(os.path.join(gt_path, sb), 'rb'))
        gt2_data = pickle.load(open(os.path.join(gt2_path, sb), 'rb'))
        tp_data = pickle.load(open(os.path.join(tp_path, sb), 'rb'))
        our_data = pickle.load(open(os.path.join(our_path, sb), 'rb'))
        print('seq_len: ', len(gt_data), 'file: ', sb)
        assert len(gt_data) == len(gt2_data)
        assert len(gt_data) == len(tp_data)
        tp_error = evaluator.eval(torch.from_numpy(gt_data), torch.from_numpy(tp_data))
        our_error = evaluator.eval(torch.from_numpy(gt_data), torch.from_numpy(our_data))
        assert len(tp_data) == len(tp_error)
        flat_dict['gt'].append(copy.copy(gt_data))
        flat_dict['tp'].append(copy.copy(tp_data))
        flat_dict['tp_err'].append(copy.copy(tp_error))
        flat_dict['our'].append(copy.copy(our_data))
        flat_dict['our_err'].append(copy.copy(our_error))
        tp_angle_error_per_sequence[ds].append(np.mean(tp_error))
        our_angle_error_per_sequence[ds].append(np.mean(our_error))
        len_per_sequence[ds].append(len(gt_data))

test = False
print('length, mean tp error, and our mean error per sequence')
for ds in datasets:
    print('for dataset: ', ds)
    for i in range(len(len_per_sequence[ds])):
        print('seq_len: {}, tp err: {}, our err: {}'.format(len_per_sequence[ds][i], tp_angle_error_per_sequence[ds][i], our_angle_error_per_sequence[ds][i]))
if test:
    exit()
for key in flat_dict.keys():
    flat_dict[key] = np.concatenate(flat_dict[key])

print('total length after flattening: ', len(flat_dict['gt']))
for key in flat_dict.keys():
    assert len(flat_dict[key]) ==  len(flat_dict['gt']) 
        

def get_10_worst(err):
    '''
    Returns the ten worst entries of all error values, along with their index in the flattened array
    '''
    sort = np.argsort(err)    
    w = []
    
    for index in sort[::-1]:
        c = 0
        for select in w:
            if np.abs(index-select) > 300:
                c+=1
        if c == len(w):
            w.append(index)
        if len(w) == 10:
            break
    return w

def get_10_best(err):
    '''
    Returns the ten best of all error values, along with their index in the flattened array
    '''
    sort = np.argsort(err)    
    b = []
    for index in sort:
        c = 0
        for select in b:
            if np.abs(index-select) > 300:
                c+=1
        if c == len(b):
            b.append(index)
        if len(b) == 10:
            break
    return b

os.makedirs('best_worst/respect_our/worst', exist_ok=True)
os.makedirs('best_worst/respect_our/best', exist_ok=True)
os.makedirs('best_worst/respect_tp/worst', exist_ok=True)
os.makedirs('best_worst/respect_tp/best', exist_ok=True)
os.makedirs('best_worst/respect_comp/worst', exist_ok=True)
os.makedirs('best_worst/respect_comp/best', exist_ok=True)

worst_our = get_10_worst(flat_dict['our_err'])
best_our = get_10_best(flat_dict['our_err'])
worst_tp = get_10_worst(flat_dict['tp_err'])
best_tp = get_10_best(flat_dict['tp_err'])
worst_comp = get_10_worst(flat_dict['tp_err'] - flat_dict['our_err'])
best_comp = get_10_best(flat_dict['tp_err'] - flat_dict['our_err'])

gtwo = flat_dict['gt'][worst_our]
gtbo = flat_dict['gt'][best_our]
gtwt = flat_dict['gt'][worst_tp]
gtbt = flat_dict['gt'][best_tp]
gtwc = flat_dict['gt'][worst_comp]
gtbc = flat_dict['gt'][best_comp]

tpwo = flat_dict['tp'][worst_our]
tpbo = flat_dict['tp'][best_our]
tpwt = flat_dict['tp'][worst_tp]
tpbt = flat_dict['tp'][best_tp]
tpwc = flat_dict['tp'][worst_comp]
tpbc = flat_dict['tp'][best_comp]

ourwo = flat_dict['our'][worst_our]
ourbo = flat_dict['our'][best_our]
ourwt = flat_dict['our'][worst_tp]
ourbt = flat_dict['our'][best_tp]
ourwc = flat_dict['our'][worst_comp]
ourbc = flat_dict['our'][best_comp]

pickle.dump(gtwo, open('best_worst/respect_our/worst/gt.p', 'wb'))
pickle.dump(tpwo, open('best_worst/respect_our/worst/tp.p', 'wb'))
pickle.dump(ourwo, open('best_worst/respect_our/worst/our.p', 'wb'))
pickle.dump(gtbo, open('best_worst/respect_our/best/gt.p', 'wb'))
pickle.dump(tpbo, open('best_worst/respect_our/best/tp.p', 'wb'))
pickle.dump(ourbo, open('best_worst/respect_our/best/our.p', 'wb'))
pickle.dump(gtwc, open('best_worst/respect_comp/worst/gt.p', 'wb'))
pickle.dump(tpwc, open('best_worst/respect_comp/worst/tp.p', 'wb'))
pickle.dump(ourwc, open('best_worst/respect_comp/worst/our.p', 'wb'))
pickle.dump(gtbc, open('best_worst/respect_comp/best/gt.p', 'wb'))
pickle.dump(tpbc, open('best_worst/respect_comp/best/tp.p', 'wb'))
pickle.dump(ourbc, open('best_worst/respect_comp/best/our.p', 'wb'))
pickle.dump(gtwt, open('best_worst/respect_tp/worst/gt.p', 'wb'))
pickle.dump(tpwt, open('best_worst/respect_tp/worst/tp.p', 'wb'))
pickle.dump(ourwt, open('best_worst/respect_tp/worst/our.p', 'wb'))
pickle.dump(gtbt, open('best_worst/respect_tp/best/gt.p', 'wb'))
pickle.dump(tpbt, open('best_worst/respect_tp/best/tp.p', 'wb'))
pickle.dump(ourbt, open('best_worst/respect_tp/best/our.p', 'wb'))

