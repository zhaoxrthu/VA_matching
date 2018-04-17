#!/usr/bin/env python

from __future__ import print_function
import os
import argparse
from optparse import OptionParser
from tools.config_tools import Config


from __future__ import division




#----------------------------------- loading paramters -------------------------------------------#
parser = OptionParser()
parser.add_option('--config',
                  type=str,
                  help="evaluation configuration",
                  default="./configs/train_config.yaml")
(opts, args) = parser.parse_args()
assert isinstance(opts, object)
opt = Config(opts.config)
print(opt)
#--------------------------------------------------------------------------------------------------#

#------------------ environment variable should be set before import torch  -----------------------#
if opt.cuda:
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
    print('setting gpu on gpuid {0}'.format(opt.gpu_id))
#--------------------------------------------------------------------------------------------------#

import random
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable

import models
from dataset import VideoFeatDataset as dset
from tools import utils






if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with \"cuda: True\"")

# setting the random seed
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
if opt.cuda and torch.cuda.is_available():
    cudnn.benchmark = True
    torch.cuda.manual_seed(opt.manualSeed)
else:
    torch.manual_seed(opt.manualSeed)
print('Random Seed: {0}'.format(opt.manualSeed))

# make checkpoint folder
if opt.checkpoint_folder is None:
    opt.checkpoint_folder = 'checkpoints'
if not os.path.exists(opt.checkpoint_folder):
    os.system('mkdir {0}'.format(opt.checkpoint_folder))

# loading dataset
train_dataset = dset(opt.data_dir, flist=opt.flist)
print('number of train samples is: {0}'.format(len(train_dataset)))
print('finished loading data')


# loading test dataset
test_video_dataset = dset(opt.data_dir, opt.video_flist, which_feat='vfeat')
test_audio_dataset = dset(opt.data_dir, opt.audio_flist, which_feat='afeat')
print('number of test samples is: {0}'.format(len(test_video_dataset)))
print('finished loading data')



# training function for metric learning
def train(train_loader, model, criterion, optimizer, epoch, opt):
    """
    train for one epoch on the training set
    """
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()

    # training mode
    model.train()

    end = time.time()
    for i, (vfeat, afeat) in enumerate(train_loader):
        # shuffling the index orders
        bz = vfeat.size()[0]
        orders = np.arange(bz).astype('int32')
        shuffle_orders = orders.copy()
        np.random.shuffle(shuffle_orders)

        # creating a new data with the shuffled indices
        afeat2 = afeat[torch.from_numpy(shuffle_orders).long()].clone()

        # concat the vfeat and afeat respectively
        afeat0 = torch.cat((afeat, afeat2), 0)
        vfeat0 = torch.cat((vfeat, vfeat), 0)

        # generating the labels
        # 1. the labels for the shuffled feats
        label1 = (orders == shuffle_orders + 0).astype('float32')
        target1 = torch.from_numpy(label1)

        # 2. the labels for the original feats
        label2 = label1.copy()
        label2[:] = 1
        target2 = torch.from_numpy(label2)

        # concat the labels together
        target = torch.cat((target2, target1), 0)
        target = 1 - target

        # put the data into Variable
        vfeat_var = Variable(vfeat0)
        afeat_var = Variable(afeat0)
        target_var = Variable(target)

        # if you have gpu, then shift data to GPU
        if opt.cuda:
            vfeat_var = vfeat_var.cuda()
            afeat_var = afeat_var.cuda()
            target_var = target_var.cuda()

        # forward, backward optimize
        sim = model(vfeat_var, afeat_var)   # inference simialrity
        loss = criterion(sim, target_var)   # compute contrastive loss

        ##############################
        # update loss in the loss meter
        ##############################
        losses.update(loss.data[0], vfeat0.size(0))

        ##############################
        # compute gradient and do sgd
        ##############################
        optimizer.zero_grad()
        loss.backward()

        ##############################
        # gradient clip stuff
        ##############################
        #utils.clip_gradient(optimizer, opt.gradient_clip)

        # update parameters
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % opt.print_freq == 0:
            log_str = 'Epoch: [{0}][{1}/{2}]\t Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t Loss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch, i, len(train_loader), batch_time=batch_time, loss=losses)
            print(log_str)

# test function for metric learning
def test(video_loader, audio_loader, model, opt):
    """
    train for one epoch on the training set
    """
    # evaluation mode: only useful for the models with batchnorm or dropout
    model.eval()

    right = 0       # correct sample number
    sample_num =0   # total sample number
    sim_mat = []    # similarity matrix

    #------------------------------------ important parameters -----------------------------------------------#
    # bz_sim: the batch similarity between two visual and auditory feature batches
    # slice_sim: the slice similarity between the visual feature batch and the whole auditory feature sets
    # sim_mat: the total simmilarity matrix between the visual and auditory feature datasets
    #-----------------------------------------------------------------------------------------------------#

    for i, vfeat in enumerate(video_loader):
        bz = vfeat.size()[0]
        sample_num += bz
        for j, afeat in enumerate(audio_loader):
            for k in np.arange(bz):
                cur_vfeat = vfeat[k].clone()
                cur_vfeats = cur_vfeat.repeat(bz, 1, 1)

                vfeat_var = Variable(cur_vfeats, volatile=True)
                afeat_var = Variable(afeat, volatile=True)
                if opt.cuda:
                    vfeat_var = vfeat_var.cuda()
                    afeat_var = afeat_var.cuda()

                cur_sim = model(vfeat_var, afeat_var)
                if k == 0:
                    bz_sim = cur_sim.clone()
                else:
                    bz_sim = torch.cat((bz_sim, cur_sim.clone()), 1)
            if j == 0:
                slice_sim = bz_sim.clone()
            else:
                slice_sim = torch.cat((slice_sim, bz_sim.clone()), 0)
        if i == 0:
            sim_mat = slice_sim.clone()
        else:
            sim_mat = torch.cat((sim_mat, slice_sim.clone()), 1)
            
    # if your metric is the feature distance, you should set descending=False, else if your metric is feature similarity, you should set descending=True
    sorted, indices = torch.sort(sim_mat, 0, descending=False)
    np_indices = indices.cpu().data.numpy()
    topk = np_indices[:opt.topk,:]
    for k in np.arange(sample_num):
        order = topk[:,k]
        if k in order:
            right = right + 1
    print('The similarity matrix: \n {}'.format(sim_mat))
    print('Testing accuracy (top{}): {:.3f}'.format(opt.topk, right/sample_num))


# learning rate adjustment function
def LR_Policy(optimizer, init_lr, policy):
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr * policy

def main():
    global opt
    # train data loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchSize,
                                     shuffle=True, num_workers=int(opt.workers))

    # create model
    if opt.model is 'VAMetric':
        model = models.VAMetric()
    elif opt.model is 'VAMetric2':
        model = models.VAMetric2()
    else:
        model = models.VAMetric()
        opt.model = 'VAMetric'

    if opt.init_model != '':
        print('loading pretrained model from {0}'.format(opt.init_model))
        model.load_state_dict(torch.load(opt.init_model))

    # Contrastive Loss
    criterion = models.ContrastiveLoss()

    if opt.cuda:
        print('shift model and criterion to GPU .. ')
        model = model.cuda()
        criterion = criterion.cuda()

    # optimizer
    optimizer = optim.SGD(model.parameters(), opt.lr,
                                momentum=opt.momentum,
                                weight_decay=opt.weight_decay)

    # adjust learning rate every lr_decay_epoch
    lambda_lr = lambda epoch: opt.lr_decay ** ((epoch + 1) // opt.lr_decay_epoch)   #poly policy

    for epoch in range(opt.max_epochs):
    	#################################
        # train for one epoch
        #################################
        train(train_loader, model, criterion, optimizer, epoch, opt)
        LR_Policy(optimizer, opt.lr, lambda_lr(epoch))      # adjust learning rate through poly policy

        ##################################
        # save checkpoints
        ##################################

        # save model every 10 epochs
        if ((epoch+1) % opt.epoch_save) == 0:
            path_checkpoint = '{0}/{1}_state_epoch{2}.pth'.format(opt.checkpoint_folder, opt.model, epoch+1)
            utils.save_checkpoint(model, path_checkpoint)

if __name__ == '__main__':
    main()













