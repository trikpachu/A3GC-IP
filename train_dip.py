
import argparse
import os
import numpy as np
import random
import torch
import torch.backends.cudnn as cudnn
from config import paths
from datasets import Dataset_dip
from tqdm import tqdm
from net_aagc import PoseNetDIP, pose_loss
import pickle

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
parser.add_argument('--hu', '--hidden_units', default=256, type=int,
                    metavar='HU', help='hidden units')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--name', dest='name',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--patience', default=3, type=int, help='early stopping patience')
parser.add_argument('--norm', action='store_true', help='normalize the data ')
parser.add_argument('--rotsize', default=9, type=int, help='size of rotation representation for target')

def train_one_epoch(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """

    model.train()

    total_len = len(train_loader)
    total_loss = 0

    bar = tqdm(enumerate(train_loader), total = total_len)
    for i, (imu, target) in bar:
        imu = imu.cuda()
        target = target.cuda()
        if args.half:
            imu = imu.half()
            target = target.half()

        # compute outputn m
        prediction, rnn_state = model.forward(imu, rnn_state=None)
        loss = criterion.forward(prediction.view(target.shape), target)                    

        bar.set_description(
                "Train[{}/{}] lr={}".format(epoch, 500, optimizer.param_groups[0]['lr']))
        bar.set_postfix_str('train loss = {}'.format(loss))
        
        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / total_len

def valid_one_epoch(val_loader, model, criterion, refine=False):
    """
    Run evaluation
    """
    # switch to evaluate mode
    model.eval()

    total_len = len(val_loader)
    total_loss = 0

    bar = tqdm(enumerate(val_loader), total = total_len)
    for i, (imu, target) in bar:
        
        imu = imu.cuda()
        target = target.cuda()
        if args.half:
            imu = imu.half()
            target = target.half()

        # compute output
        with torch.no_grad():
            prediction, rnn_state = model.forward(imu, rnn_state=None)
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
    #setup_seed(3407)
    global args
    args = parser.parse_args()
    for arg in vars(args):
        print(arg, '==' ,getattr(args, arg))

    start_epoch = 0

    model_save_dir = os.path.join('trained_models', args.name)
    # Check the save_dir exists or not
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    device = torch.device("cuda:0")
    model = PoseNetDIP(rotsize=args.rotsize, n_hidden=args.hu, device=device).to(device)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            if args.finetuning:
                start_epoch = 0
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            raise ValueError('You wanted me to load a checkpoint, but I could not')
    
    cudnn.benchmark = True

    train_dataset = Dataset_dip(os.path.join(paths.amass_dir if not args.finetuning else paths.dipimu_dir, "train_tp_global_chunked.pt"), rotsize=args.rotsize, norm=args.norm)
    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    if args.fse:
        val_dataset = Dataset_dip(os.path.join(paths.amass_dir if not args.finetuning else paths.dipimu_dir, "valid_tp_global.pt"), rotsize=args.rotsize, norm=args.norm)
        val_loader = torch.utils.data.DataLoader(val_dataset,
            batch_size=1, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    else:
        val_dataset = Dataset_dip(os.path.join(paths.amass_dir if not args.finetuning else paths.dipimu_dir, "valid_tp_global_chunked.pt"), rotsize=args.rotsize, norm=args.norm)
        val_loader = torch.utils.data.DataLoader(val_dataset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    print(f"train len {len(train_loader)}, valid len {len(val_loader)}")
    # 
    criterion = pose_loss()

    if args.half:
        model.half()
        criterion.half()
    
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    best_loss = 1e5
    tolerance_counter = 0

    for epoch in range(start_epoch, 500):
        # adjust_learning_rate(optimizers, epoch)
        # train for one epoch
        train_loss = train_one_epoch(train_loader, model, criterion, optimizer, epoch)
        scheduler.step()
        
        # evaluate on validation set
        valid_loss = valid_one_epoch(val_loader, model, criterion)
        print('|---------- epoch = {}  |  train_loss = {}  |  valid_loss = {} ----------|'.format(epoch, train_loss, valid_loss))
        
        if valid_loss < best_loss:
            tolerance_counter = 0
            best_loss = valid_loss
            save_checkpoint({ 'epoch': epoch + 1, 'state_dict': model.state_dict(),}, 
                            filename=os.path.join(model_save_dir, 'checkpoint_{}_{}.tar'.format("finetuning" if args.finetuning else "pretrain", epoch)))
        else:
            tolerance_counter += 1
        if tolerance_counter > args.patience:
            break


if __name__ == '__main__':
    main()