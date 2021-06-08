from __future__ import print_function
#%matplotlib inline
import argparse
import os
import time
import datetime
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.autograd as autograd
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import math
import numpy as np
import matplotlib.pyplot as plt

import matplotlib
import torchvision
from src.utils import *
import src.losses as losses
import torch.nn.functional as F
from anom_utils import post_process, generate_image, reconstruction_loss, latent_reconstruction_loss
from anom_utils import l1_latent_reconstruction_loss, anomaly_score, score_and_auc, score_and_auc_sep, ood_aucs
from torchvision.utils import save_image
from torch.optim.lr_scheduler import MultiStepLR
import wandb

import data
from newevaluate import evaluate
from cifar_models_bNorm import Encoder,Res_Discriminator, ResnetDiscriminator32, ResnetGenerator32

matplotlib.rc("text", usetex=False)



parser = argparse.ArgumentParser()

parser.add_argument('--manualSeed', type = int, default=5, help='set the seed for the model manually')
parser.add_argument('--dataroot',  default = "./data/CIFAR10/",
                help='Location of the data')
parser.add_argument('--dataset', default='cifar10', help='name of the dataset we are working with')
parser.add_argument('--anom_pc',type = float ,default = 0, help='percentage of each anomaly class to use in training')
parser.add_argument('--batchsize', type = int, default=256, help='train/val batchsize')
parser.add_argument('--val_split', type = float, default=0.05, help='%of train data to split into val data')

parser.add_argument('--image_size', type = int, default= 32, help='size of training images')
parser.add_argument('--num_channels', type = int, default=3, help='number of channels')

parser.add_argument('--ngpu', type = int, default=1, help='number of GPUs available')
parser.add_argument('--workers', type = int, default=4, help='number of worker CPU nodes')
parser.add_argument('--abnormal_classes', default=None, help='name of the abnormal class, changes based on the dataset')
parser.add_argument('--normal_class', default=None, help='name of the normal class, changes based on the dataset')

parser.add_argument('--lr',type = float ,default = 3e-4)
parser.add_argument('--nz',type = int ,default = 128)
parser.add_argument('--model_load_path',default = '', help='path to trained model, otherwise the model will train from scratch')
parser.add_argument('--cuda',type= int, default = 0)
parser.add_argument('--recon_loss_type',type = str, default = 'wasserstein', help = 'loss type to be used in reconstruction discriminator')
parser.add_argument('--ae_recon_loss_type',type = str, default = 'wasserstein', help = 'loss type to be used in AE update')


parser.add_argument('--num_epochs',type = int, default = 100, help = 'number of training epochs')
parser.add_argument('--start', type=int, default = 0)
parser.add_argument('--use_penalty', action = 'store_true')
parser.add_argument('--ch', type=int,default = 128)
parser.add_argument('--update_ratio', type=int,default = 5, help='number of discriminator updates to generator update')
parser.add_argument('--save_model_epochs',type = int, default = 30, help='save the model after this many epochs')
parser.add_argument('--save_logs_epochs',type = int, default = 1, help='save the model after this many epochs')
parser.add_argument('--save_model_root',  default = "logs",
                help='Location to save the model wrt train file')
parser.add_argument('--anom_recon_lambda',type = float, default = 1, help='anomaly loss reconstruction hyperparameter')
parser.add_argument('--regularizer_lambda',type = float, default = 1, help='regularization term hyperparameter')
parser.add_argument('--gp_lambda',type = float, default = 10, help='gradient penalty hyperparameter')
parser.add_argument('--spectral_norm', type=int, default = 0 , help='whether to use spectral norm in reconstruction Discriminator')

parser.add_argument('--anom_metric_type', type=str, default = 'f_anomloss', help='anomloss|l1_mean_recon')
parser.add_argument('--augment', action = 'store_true')
parser.add_argument('--ood_model', action = 'store_true')
parser.add_argument('--learning_setting', type=str, default = 'ss', help='ss|unsupervised - if unsupervised that anompc fraction of anomalies get mistagged as normals')
parser.add_argument('--interpolation_in_recon', action = 'store_true')
parser.add_argument('--expt_name', type=str, default = 'temp', help='Any specific name to the expt?')
parser.add_argument('--corruption',type = float, default = 0, help='percentage corruption of anomalous labels in training data')
parser.add_argument('--ip_lambda',type = float, default = 0.2, help='interpolation hyperparameter hyperparameter')
parser.add_argument('--sampling', action = 'store_true')
parser.add_argument('--corrup', action = 'store_true')
parser.add_argument('--atyp_selec_style', type=str, default = 'inward', help='inward|outward|sipple')
parser.add_argument('--atyp_ratio', type=float, default = 0.5, help='How far from the annulus we want to sample for anomalies')




opt = parser.parse_args()


if opt.interpolation_in_recon:
    from interpolation import recon_discriminator_loss, recon_loss_joint
    rc = 'interpol_'
    
else:
    from loss import recon_discriminator_loss, recon_loss_joint
    rc=''

date = str(datetime.datetime.now())
date = date[:date.rfind(":")].replace("-", "")\
                                     .replace(":", "")\
                                     .replace(" ", "_")
        
if opt.augment:
    ag = 'augment_'
else:
    ag=''
    
if opt.ood_model:
    cf = 'ood'
else:
    cf = opt.normal_class
runname = str(opt.expt_name+'_' + rc + cf) +  '_' + str(opt.anom_recon_lambda)+ 'anomlamb_'+ date

if opt.dataset == 'cifar10':
    wandb.init(project="cifar10", name=runname, group=opt.expt_name)
    model_check_epoch = 20
    all_classes = ['airplane','automobile','bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
else:
    print('Unknown dataset!')
    raise
wandb.config.update(opt)

print("Chosen Seed: ", opt.manualSeed)
torch.manual_seed(opt.manualSeed)
np.random.seed(opt.manualSeed)


#size of generator feature maps
ngf = opt.image_size

#size of discriminator feature maps
ndf = opt.image_size

#num of GPUs available
ngpu = opt.ngpu

#Number of discriminator updates per generator update
tfd = opt.update_ratio

# save images, histograms every __ epochs
save_rate_logs = opt.save_logs_epochs
save_rate_model = opt.save_model_epochs
best_auc = 0
best_epoch=1
best_anomscore = 10e10


print(opt)



#loading the data

b_size = opt.batchsize
if opt.abnormal_classes is not None:
    anom_classes = eval(opt.abnormal_classes)
elif opt.normal_class is not None:
    norm_class = eval(opt.normal_class)
    anom_classes = list(set(all_classes) - set(norm_class))
else:
    anom_classes = None
    print('We are in OOD expt')

dataset_train, dataset_valid, dataset_test = data.load_data(dataset = opt.dataset, ood = opt.ood_model, anom_classes = anom_classes,anom_ratio=opt.anom_pc,corruption= opt.corruption, seed = opt.manualSeed, augmentation = opt.augment, learning_setting= opt.learning_setting, valid_split = opt.val_split)



trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=b_size, 
                                          drop_last = True,
                                           shuffle = True, num_workers=opt.workers)
validloader = torch.utils.data.DataLoader(dataset_valid, batch_size=b_size, 
                                           shuffle = False, num_workers=opt.workers)
testloader = torch.utils.data.DataLoader(dataset_test, batch_size=b_size, 
                                           shuffle = False, num_workers=opt.workers)

lr = opt.lr

##creating folders to save the models and logs

imgroot = os.path.join('newlogs',runname, 'image' )
saveModelRoot = os.path.join('newlogs',runname, 'model' )
print(imgroot)
print(saveModelRoot)

try:
    os.makedirs(saveModelRoot)
except OSError: 
    print("Model folder already exists")

try:
    os.makedirs(imgroot)
except OSError: 
    print_function("Image folder already exists")

#set the device
device = torch.device("cuda:%s" % (opt.cuda) if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)

#define the models
# netE is the encoder, netG is the Generator and netD2 is the reconstruction discriminator
netE = Encoder(ngpu=1,nz=opt.nz, nc = opt.num_channels,ndf = ndf)
netG = ResnetGenerator32(z_dim = opt.nz)
netD2 = ResnetDiscriminator32(stack = 2*opt.num_channels , ch= opt.ch, spectral_norm=opt.spectral_norm)

    

netE.to(device)
netG.to(device)
if netD2 is not None:
    netD2.to(device)
    optimizerD2 = optim.Adam(netD2.parameters(), lr = lr, betas = (0, .9))
optimizerG = optim.Adam(netG.parameters(), lr = lr, betas = (0, .9))
optimizerE = optim.Adam(netE.parameters(), lr = lr, betas = (.5, .9))


schedulerD2 = MultiStepLR(optimizerD2, milestones=[30,60,90], gamma=0.1)
schedulerG = MultiStepLR(optimizerG, milestones=[30,60,90], gamma=0.1)
schedulerE = MultiStepLR(optimizerE, milestones=[30,60,90], gamma=0.1)


start = opt.start
print("Starting Training")

print('start from %s ' % start)
n_iter = 0

starttime = time.time()

if(opt.model_load_path==''):
    for epoch in range(start,opt.num_epochs):
                  
        netE.train()
        if netD2 is not None:
            netD2.train()
        netG.train()
 
        for i, data in enumerate(trainloader, 0):
            n_iter = n_iter + 1
            imgs = data[0].to(device)
            labels = data[1].to(device)
            real_idx =  ((labels > 0).nonzero()).squeeze(1)
            anom_idx =  ((labels < 0).nonzero()).squeeze(1)
            if opt.corrup:
                labels[anom_idx] = 1
            if opt.anom_pc==0 or opt.corrup:
                if opt.sampling and epoch > 10:
                    netE.eval() ; netG.eval()
                    anom_count = int(0.1*len(labels))
                    anom_idx = np.random.choice(range(len(labels)), anom_count)
                    labels[anom_idx] = -1
                    alpha = torch.FloatTensor(anom_count, 1).uniform_(0, opt.atyp_ratio).to(device)
                    z = torch.zeros([anom_count, opt.nz], dtype=torch.float32, device=device )
                    with torch.no_grad():
                        if opt.atyp_selec_style == 'inward':
                            e = (1-alpha)*z + (alpha)*netE(imgs[anom_idx]).detach()
                        elif opt.atyp_selec_style == 'outward':
                            k = netE(imgs[anom_idx]).detach()
                            e = (k - (1-alpha)*z)/alpha
                        elif opt.atyp_selec_style == 'sipple':
                            e = torch.Tensor(torch.rand(anom_count,opt.nz)*(torch.min(torch.abs(netE(imgs))))).to(device).detach()
                        else:
                            print('Unknown Negative Sampling style!')
                            raise 
                        imgs[anom_idx] = netG(e).detach()
                    netE.train() ; netG.train()
            else:
                real = imgs[real_idx]
                anom = imgs[anom_idx]
                real_labels = labels[real_idx]

            # optimize discriminator tfd times
            for t in range(tfd):
                netE.zero_grad()
                netG.zero_grad()

                if netD2 is not None:
                    netD2.zero_grad()
                    netE.zero_grad()
                    netG.zero_grad()
                    errD_recon = recon_discriminator_loss(imgs,labels, netD2, netE, netG,losstype =opt.recon_loss_type,anom_lambda=opt.anom_recon_lambda, use_penalty=opt.use_penalty, gradp_lambda = opt.gp_lambda,ip_lambda=opt.ip_lambda)
                    errD_recon.backward()
                    optimizerD2.step()
                else:
                    errD_recon = 0
                    
            reg_term = (torch.norm(netE(imgs[labels==1]), dim = (1))).mean()            
            if netD2 is not None:             
                recon_loss_encoded = recon_loss_joint(imgs, labels,netG, netE, netD2, loss_type = opt.ae_recon_loss_type,anom_lambda=opt.anom_recon_lambda,ip_lambda=opt.ip_lambda )
            else:
                recon_loss_encoded = recon_loss_joint(imgs, labels,netG, netE, netD2, loss_type = 'l1',
                                                      anom_lambda=opt.anom_recon_lambda,ip_lambda=opt.ip_lambda )
            errG =  recon_loss_encoded  +  opt.regularizer_lambda*reg_term 
  
            netG.zero_grad()
            netE.zero_grad()
            if netD2 is not None:
                netD2.zero_grad()
            
            errG.backward()
            optimizerG.step()
            optimizerE.step()

            wb_iter = len(trainloader)*epoch + n_iter
            wandb.log({'epoch': epoch,'iteration':wb_iter ,
                          'loss_Drecon':errD_recon, 'loss_AE': errG, 'reconloss_inAE':recon_loss_encoded, 'reg_term':reg_term})
            if i % 50 == 0:
                print('[%d/%d][%d/%d]'
                    %(epoch, opt.num_epochs, i, len(trainloader)))
                
                

    ### validating and saving area ###

        if epoch % save_rate_logs == 0:
            netE.eval()
            if netD2 is not None:
                netD2.eval()
            netG.eval()

            with torch.no_grad():

                # calculating auroc,auprc on test dataset. test_norm_auc is the auc according to proposed anomaly score. 
                _, overall_test_score, real_test_score, anom_test_score, test_auprc, test_dis_auc,test_norm_auc = score_and_auc_sep(testloader,trainloader ,netG, netE, netD2,device ,ngpu, break_iters = 10000,anom_metric_type = opt.anom_metric_type)
                print(('test_auc:%f') % test_norm_auc)

                _, val_anom_score,_,_, valid_auprc, valid_dis_auc,valid_norm_auc = score_and_auc_sep(validloader,trainloader, netG, netE, netD2, device,ngpu, break_iters = 50,anom_metric_type= opt.anom_metric_type)
                print(('train_score:%f') % val_anom_score)

                wandb.log({'overall_test_score':overall_test_score, 
                           'real_test_score':real_test_score, 'anom_test_score':anom_test_score ,
                           'valid_anom_mean': val_anom_score, 'test_dis_auc': test_dis_auc,
                          'valid_dis_auc' : valid_dis_auc, 'test_norm_auc':test_norm_auc,
                          'valid_norm_auc':valid_norm_auc})
                
                with open(os.path.join(imgroot, "epoch_losses.txt"), "a") as f:
                    currenttime = time.time()
                    elapsed = currenttime - starttime
                    f.write("{} \t {:.2f}\t {:.5f}\t {:.5f}\t {:.5f}\t {:.5f}".format(epoch, elapsed, test_norm_auc, test_auprc, overall_test_score,valid_norm_auc, valid_auprc ,val_anom_score) + "\n")
                starttime = time.time()

                   # save images
                output = None
                for i,data in enumerate(testloader, 0):
                    targets = data[1].to(device)
                    test = data[0][:64,:,:,:].to(device)
                    row = torch.cat((test,netG(netE(test))), dim=2)
                    if output is None:
                        output = row
                    else:
                        output = torch.cat((output, row), dim=1)
                    save_image(output, os.path.join(imgroot, "img-{}.png".format(epoch)))
                    wandb.log({"reconstructions": [wandb.Image(output, caption="{}".format(epoch))]})

                    break
                    best_anomscore
                if epoch >= model_check_epoch: 
                    if valid_norm_auc > best_auc:
                        best_auc = valid_norm_auc
                        best_epoch = epoch
                        best_model = True
                    else:
                        best_model = False
                    
                if epoch>= model_check_epoch and best_model:
                    wandb.run.summary["best_valid_auc"] = valid_norm_auc
                    wandb.run.summary["best_epoch"] = epoch
                    wandb.run.summary["test_auc_atbestvalid"] = test_norm_auc
                    print('Best_model_epoch:%d, Best val auc:%f' %(best_epoch,best_auc))

                    
                    torch.save(netG.state_dict(),'%s/netbestG.pth' % (saveModelRoot))
                    torch.save(netE.state_dict(),'%s/netbestE.pth' % (saveModelRoot))
                    if netD2 is not None:
                        torch.save(netD2.state_dict(),'%s/netbestD2.pth' % (saveModelRoot))
                    
                if epoch % save_rate_model == 0:
                    torch.save(netG.state_dict(),'%s/netG_%s.pth' % (saveModelRoot, epoch))
                    torch.save(netE.state_dict(),'%s/netE_%s.pth' % (saveModelRoot,epoch))
                    if netD2 is not None:
                        torch.save(netD2.state_dict(),'%s/netD2_%s.pth' % (saveModelRoot,epoch))
                        
        schedulerD2.step()
        schedulerG.step()
        schedulerE.step()
        
