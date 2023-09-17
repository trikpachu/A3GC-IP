
import argparse
import os
import numpy as np
import random
import torch
import torch.backends.cudnn as cudnn
from config import paths
from datasets import Dataset_tp
from tqdm import tqdm
from net_aagc import PoseNetTP, pose_loss
import pickle
import glob 
import re 

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--batch_size",metavar="x", type=int, required=True)
parser.add_argument("--fse", "--full_sequence_validation",action="store_true", help="validate on full sequences")
parser.add_argument("-f", "--finetuning", action="store_true", help="isFineTuning")
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--hu1', '--hidden_units1', default=256, type=int,
                    metavar='HU1', help='hidden units1')
parser.add_argument('--hu2', '--hidden_units2', default=64, type=int,
                    metavar='HU2', help='hidden units2')
parser.add_argument('--hu3', '--hidden_units3', default=128, type=int,
                    metavar='HU3', help='hidden units3')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--name', dest='name',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--start_at', default=1, type=int, help='at which model to start training (from scratch), since the code trains three models')
parser.add_argument('--rotsize', default=6, type=int, help='size of rotation representation for target')
parser.add_argument('--norm', action='store_true', help='normalize the data ')
parser.add_argument('--ankle', action='store_true', help='using the synthetic dataset with sensors on the ankle rather than below the knee ')
parser.add_argument('--gpu_index', type=int, default=0, help='which GPU to use ')
parser.add_argument('--patience', type=int, default=3, help='which GPU to use ')
parser.add_argument('--no_common_stats', action='store_true', help='dont use the stats of the training_data')
parser.add_argument('--all_stats', action='store_true', help='dont use the stats of the training_data')
parser.add_argument('--cda', action='store_true', help='use cda ')

def train_one_epoch(train_loader, model, criterion, optimizer, epoch, model_number = 1):
    """
        Run one train epoch
    """

    model.train()

    total_len = len(train_loader)
    total_loss = 0

    bar = tqdm(enumerate(train_loader), total = total_len)
    for i, (imu, leaf_pos_input, full_pos_input, leaf_pos, full_pos, smpl) in bar:
        if model_number == 1:
            inputs = imu
            target = leaf_pos
        elif model_number == 2:
            inputs = torch.cat((imu, leaf_pos_input), dim=-1)
            target = full_pos
        elif model_number == 3:
            inputs = torch.cat((imu, full_pos_input), dim=-1)
            target = smpl

        inputs = inputs.cuda()
        target = target.cuda()
        if args.half:
            inputs = inputs.half()
            target = target.half()

        # compute outputn m
        prediction, rnn_state = model.forward(inputs, rnn_state=None)
        loss = criterion.forward(prediction.view(target.shape), target)                    

        bar.set_description(
                "Model {} Train[{}/{}] lr={}".format(model_number, epoch, 500, optimizer.param_groups[0]['lr']))
        bar.set_postfix_str('train loss = {}'.format(loss))
        
        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / total_len

def valid_one_epoch(val_loader, model, criterion, refine=False, model_number=1):
    """
    Run evaluation
    """
    # switch to evaluate mode
    model.eval()

    total_len = len(val_loader)
    total_loss = 0

    bar = tqdm(enumerate(val_loader), total = total_len)
    for i, (imu, leaf_pos_input, full_pos_input, leaf_pos, full_pos, smpl) in bar:
        if model_number == 1:
            inputs = imu
            target = leaf_pos
        elif model_number == 2:
            inputs = torch.cat((imu, leaf_pos), dim=-1)
            target = full_pos
        elif model_number == 3:
            inputs = torch.cat((imu, full_pos), dim=-1)
            target = smpl

        inputs = inputs.cuda()
        target = target.cuda()
        if args.half:
            inputs = inputs.half()
            target = target.half()

        # compute output
        with torch.no_grad():
            prediction, rnn_state = model.forward(inputs, rnn_state=None)
            loss = criterion.forward(prediction.view(target.shape), target)
        total_loss += loss.item()
        bar.set_description("Valid")
        bar.set_postfix_str('valid loss = {}'.format(loss))

    return total_loss / total_len

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def main():
    setup_seed(23)
    global args
    args = parser.parse_args()
    for arg in vars(args):
        print(arg, '==' ,getattr(args, arg))

    start_epoch = 0

    model_save_dir = os.path.join('trained_models', args.name)
    # Check the save_dir exists or not
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    with open('./nira_template_15_norm.pkl', 'rb') as f:
        nira = pickle.load(f) 
    nira = torch.from_numpy(nira).float()

    device = torch.device("cuda:0")
    model1 = PoseNetTP(input_size=72, n_output=15, n_hidden=256, device=device).to(device)
    model2 = PoseNetTP(input_size=87, n_output=69, n_hidden=64, device=device).to(device)
    model3 = PoseNetTP(input_size=141, n_output=15*args.rotsize, n_hidden=128, device=device).to(device)

    # optionally resume from a checkpoint
    if args.resume:
        model_path = args.resume
        model_list = glob.glob(os.path.join(model_path, '*'))
        model_list = [i for i in model_list if 'pretrain' in i]
        model1_file = [i for i in model_list if 'model1' in i]
        model2_file = [i for i in model_list if 'model2' in i]
        model3_file = [i for i in model_list if 'model3' in i]
        model1_file = list(map(list, zip(*[[i, int(re.findall('_\d+', os.path.basename(i))[0][1:])] for i in model1_file])))
        model2_file = list(map(list, zip(*[[i, int(re.findall('_\d+', os.path.basename(i))[0][1:])] for i in model2_file])))
        model3_file = list(map(list, zip(*[[i, int(re.findall('_\d+', os.path.basename(i))[0][1:])] for i in model3_file])))
        model1_file = model1_file[0][np.argmax(model1_file[1])]
        model2_file = model2_file[0][np.argmax(model2_file[1])]
        model3_file = model3_file[0][np.argmax(model3_file[1])]
        print('Loading the following checkpoints: ')
        print(model1_file)
        print(model2_file)
        print(model3_file)
        checkpoint1 = torch.load(model1_file)
        checkpoint2 = torch.load(model2_file)
        checkpoint3 = torch.load(model3_file)
        model1.load_state_dict(checkpoint1['state_dict'])
        model2.load_state_dict(checkpoint2['state_dict'])
        model3.load_state_dict(checkpoint3['state_dict'])
    
    cudnn.benchmark = True

    if args.ankle:
        amass_path = paths.amass_dir_ankle
        dip_path = paths.dipimu_dir
    elif args.cda:
        amass_path = paths.amass_dir_sym
        dip_path = paths.dipimu_dir_sym
    else:
        amass_path = paths.amass_dir
        dip_path = paths.dipimu_dir

    train_dataset = Dataset_tp(os.path.join(amass_path if not args.finetuning else dip_path, "train_tp_global_chunked_pos.pt"), rotsize=args.rotsize, norm=args.norm, ankle=args.ankle, sym=args.cda)
    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    if args.fse:
        val_dataset = Dataset_tp(os.path.join(amass_path if not args.finetuning else dip_path, "valid_tp_global_pos.pt"), rotsize=args.rotsize, norm=args.norm, ankle=args.ankle, sym=args.cda)
        val_loader = torch.utils.data.DataLoader(val_dataset,
            batch_size=1, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    else:
        val_dataset = Dataset_tp(os.path.join(amass_path if not args.finetuning else dip_path, "valid_tp_global_chunked_pos.pt"), rotsize=args.rotsize, norm=args.norm, ankle=args.ankle, sym=args.cda)
        val_loader = torch.utils.data.DataLoader(val_dataset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    print(f"train len {len(train_loader)}, valid len {len(val_loader)}")
    # 
    criterion1 = pose_loss()
    criterion2 = pose_loss()
    criterion3 = pose_loss()

    if args.half:
        model.half()
        criterion.half()
    
    optimizer1 = torch.optim.Adam(model1.parameters(), args.lr, weight_decay=args.weight_decay)
    optimizer2 = torch.optim.Adam(model2.parameters(), args.lr, weight_decay=args.weight_decay)
    optimizer3 = torch.optim.Adam(model3.parameters(), args.lr, weight_decay=args.weight_decay)
    scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer1, gamma=0.8)
    scheduler2 = torch.optim.lr_scheduler.ExponentialLR(optimizer2, gamma=0.8)
    scheduler3 = torch.optim.lr_scheduler.ExponentialLR(optimizer3, gamma=0.8)

    if args.start_at == 1:
        best_loss = 1e5
        tolerance_counter = 0

        for epoch in range(start_epoch, 500):
            # train for one epoch
            train_loss = train_one_epoch(train_loader, model1, criterion1, optimizer1, epoch, model_number=1)
            scheduler1.step()
            
            # evaluate on validation set
            valid_loss = valid_one_epoch(val_loader, model1, criterion1, model_number=1)
            print('|---------- epoch = {}  |  train_loss = {}  |  valid_loss = {} ----------|'.format(epoch, train_loss, valid_loss))
            
            if valid_loss < best_loss:
                tolerance_counter = 0
                best_loss = valid_loss
                save_checkpoint({ 'epoch': epoch + 1, 'state_dict': model1.state_dict(),}, 
                                filename=os.path.join(model_save_dir, 'checkpoint_model1_{}_{}.tar'.format("finetuning" if args.finetuning else "pretrain", epoch)))
            else:
                tolerance_counter += 1
            if tolerance_counter > args.patience:
                break

    if args.start_at <= 2:
        best_loss = 1e5
        tolerance_counter = 0

        for epoch in range(start_epoch, 500):
            # train for one epoch
            train_loss = train_one_epoch(train_loader, model2, criterion2, optimizer2, epoch, model_number=2)
            scheduler2.step()
            
            # evaluate on validation set
            valid_loss = valid_one_epoch(val_loader, model2, criterion2, model_number=2)
            print('|---------- epoch = {}  |  train_loss = {}  |  valid_loss = {} ----------|'.format(epoch, train_loss, valid_loss))
            
            if valid_loss < best_loss:
                tolerance_counter = 0
                best_loss = valid_loss
                save_checkpoint({ 'epoch': epoch + 1, 'state_dict': model2.state_dict(),}, 
                                filename=os.path.join(model_save_dir, 'checkpoint_model2_{}_{}.tar'.format("finetuning" if args.finetuning else "pretrain", epoch)))
            else:
                tolerance_counter += 1
            if tolerance_counter > args.patience:
                break

    if args.start_at <= 3:
        best_loss = 1e5
        tolerance_counter = 0

        for epoch in range(start_epoch, 500):
            # train for one epoch
            train_loss = train_one_epoch(train_loader, model3, criterion3, optimizer3, epoch, model_number=3)
            scheduler3.step()
            
            # evaluate on validation set
            valid_loss = valid_one_epoch(val_loader, model3, criterion3, model_number=3)
            print('|---------- epoch = {}  |  train_loss = {}  |  valid_loss = {} ----------|'.format(epoch, train_loss, valid_loss))
            
            if valid_loss < best_loss:
                tolerance_counter = 0
                best_loss = valid_loss
                save_checkpoint({ 'epoch': epoch + 1, 'state_dict': model3.state_dict(),}, 
                                filename=os.path.join(model_save_dir, 'checkpoint_model3_{}_{}.tar'.format("finetuning" if args.finetuning else "pretrain", epoch)))
            else:
                tolerance_counter += 1
            if tolerance_counter > args.patience:
                break

        

if __name__ == '__main__':
    main()