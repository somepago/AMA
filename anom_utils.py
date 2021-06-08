import torch
import torchvision.transforms as transforms
from newevaluate import evaluate, auprc
from itertools import chain
import numpy as np
from collections import Counter
import torch.nn as nn
from torchvision.datasets import SVHN, CIFAR10, ImageFolder
import os
from numpy import linalg as LA
from scipy.stats import norm



def post_process(image):
	image = image.view(-1, 3, 32, 32)
	image = image.mul(0.5).add(0.5)
	return image

def generate_image(image, frame, name):
	image = image.cpu()
	image = post_process(image)
	image = transforms.ToPILImage()(vutils.make_grid(image, padding=2, normalize=False))


def reconstruction_loss(image1, image2):
	nc, image_size, _ = image1.shape
	image1, image2 = post_process(image1), post_process(image2)
	norm = torch.norm((image2 - image1).view(-1,nc*image_size*image_size), dim=(1))
	return norm.view(-1).data.cpu().numpy()


#Calculates the L2 loss between image1 and image2
def latent_reconstruction_loss(image1, image2):
	norm = torch.norm((image2 - image1), dim=1)
	return norm.view(-1).data.cpu().numpy()

def l1_latent_reconstruction_loss(image1, image2):
	norm = torch.sum(torch.abs(image2 - image1),dim=1)
	return norm.view(-1).data.cpu().numpy()

    

def adjust_learning_rate(optimizer, epoch, num_epochs, lrate):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lrate - lrate * (epoch-45)/(num_epochs - 45)
    print('use learning rate %f' % lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def half_adjust_learning_rate(optimizer, epoch, num_epochs, lrate):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lrate - 1e-4
    print('use learning rate %f' % lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
        
def anomaly_score(data, netG, netE, netD2, ngpu=1,anom_metric_type='f_anomloss'):
    
	if netD2 is None:
		anom_metric_type = 'l1_mean_recon'
	if anom_metric_type == 'l1_mean_recon':
		return torch.mean(torch.abs(data - netG(netE(data))).view(data.shape[0],-1),dim=1).detach().cpu()
    
	if (ngpu > 1):
		a1 = netD2.module.feature(torch.cat((data,data),dim =1))
		a2 = netD2.module.feature(torch.cat((data,netG(netE(data)).detach()),dim=1))
	else:
		a1 = netD2.feature(torch.cat((data,data),dim =1))
		a2 = netD2.feature(torch.cat((data,netG(netE(data)).detach()),dim=1))
	return l1_latent_reconstruction_loss(a1,a2)



def score_and_auc(dataLoader, netG, netE, netD2, device, ngpu=1, break_iters = 10,anom_metric_type='f_anomloss'):
	score_list = []
	score_label = []
	count=0
	with torch.no_grad():
		for i,data in enumerate(dataLoader, 0):
			if count>=break_iters:
				break
			real = data[0].to(device)
			score_label.append(data[1].to(device).tolist())
			score_list.append(anomaly_score(real,netG, netE, netD2, ngpu, anom_metric_type))
			
			count+=1
		score_list = list(chain.from_iterable(score_list))
		score_label = list(chain.from_iterable(score_label))
# 		print(Counter(score_label))
		if Counter(score_label)[-1] == 0:
			score_auc = 0
		else:
			score_auc = 1- evaluate(score_label,score_list)
		score_anom_mean = np.array(score_list).mean()
	return score_auc, score_anom_mean

def score_and_auc_sep(dataLoader, trainloader, netG, netE, netD2, device, ngpu=1, break_iters = 10,anom_metric_type='f_anomloss'):
    score_list = []
    score_label = []
    disn_list = []
    count=0
    
    with torch.no_grad():
        train_score_list = np.array([])
        for i,data in enumerate(trainloader, 0):
            real = data[0].to(device) 
            train_score_list = np.append(train_score_list,anomaly_score(real,netG, netE, netD2))  
#             print(i)

        for i,data in enumerate(dataLoader, 0):
            real = data[0].to(device)
            score_label.append(data[1].to(device).tolist())
            score_list.append(anomaly_score(real,netG, netE, netD2, ngpu,anom_metric_type)) 
            disn_list.append(LA.norm(netE(real).detach().cpu().numpy(),axis=1))
        score_list = list(chain.from_iterable(score_list))
        score_label = list(chain.from_iterable(score_label))
        disn_list = list(chain.from_iterable(disn_list))
#         import ipdb; ipdb.set_trace()
        mu, std = norm.fit(train_score_list)
        predictions = []
        for s in score_list:
            p = norm(mu, std).pdf(s)
            predictions.append(p)
        norm_auc =  evaluate(score_label,predictions)  
        
        score_mean = np.median(np.array(score_list))
        
        if -1 not in score_label:
            return 0, score_mean, 0, 0, 0
        score_auc = 1- evaluate(score_label,score_list)
        dis_auc = 1- evaluate(score_label,disn_list)
        prc = evaluate(score_label,score_list,metric = 'auprc')
        real_idx =  np.array(score_label)>0
        anom_idx =  np.array(score_label)<0   
        real_mean = np.median(np.array(score_list)[real_idx])
        anom_mean = np.median(np.array(score_list)[anom_idx])
        
    return score_auc, score_mean, real_mean, anom_mean, prc,dis_auc, norm_auc




def only_auc(dataLoader1, dataloader2,train_score_list, netG, netE, netD2,device, ngpu=1, break_iters = 10,anom_metric_type='f_anomloss'):
    score_list = []
    score_label = []
    count=0
    with torch.no_grad():
        for i,data in enumerate(dataLoader1, 0):
            real = data[0].to(device)
            score_label.append(data[1].to(device).tolist())
            score_list.append(anomaly_score(real,netG, netE, netD2, ngpu,anom_metric_type))    
        max_i = i
        count = 0
        for i,data in enumerate(dataloader2, 0):
#             print(i)
            if count > max_i//5:
                continue
            count+=1
            real = data[0].to(device)
            score_label.append(data[1].to(device).tolist())
            score_list.append(anomaly_score(real,netG, netE, netD2, ngpu,anom_metric_type))  
            
        score_list = list(chain.from_iterable(score_list))
        score_label = list(chain.from_iterable(score_label))
        
        score_auc = 1- evaluate(score_label,score_list)
        
        mu, std = norm.fit(train_score_list)
        predictions = []
        for s in score_list:
            p = norm(mu, std).pdf(s)
            predictions.append(p)
        norm_auc =  evaluate(score_label,predictions) 
        
    return score_auc, norm_auc


datalocations = {
    'icons':'./data/Icons-50/',
    'lsun':'./data/LSUN_resize/',
    'isun':'./data/iSUN/',
    'imagenet':'./data/Imagenet_resize/',
    'textures':'./data/dtd/images/'
}



def ood_aucs(trainloader, netG, netE, netD2,wandb, imgroot,epoch, device):
    aucs = []
    transform = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )
    transform_icons = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )
    cifar_test = CIFAR10('./data/CIFAR10', train=False, transform=transform, target_transform=lambda x:1, download=False)
    testloader = torch.utils.data.DataLoader(cifar_test, batch_size=64, 
                                         drop_last = True,
                                           shuffle = False, num_workers=4)
    
#     svhn_test = SVHN(root= './data/SVHN',transform=transform, download=True,split='test',target_transform=lambda x:1)
#     testloader = torch.utils.data.DataLoader(svhn_test, batch_size=64, 
#                                          drop_last = True,
#                                            shuffle = False, num_workers=4)
    
    
#     for dataset in ['lsun','isun','imagenet','textures']:

    with torch.no_grad():
        train_score_list = np.array([])
        for i,data in enumerate(trainloader, 0):
            real = data[0].to(device) 
            train_score_list = np.append(train_score_list,anomaly_score(real,netG, netE, netD2))  

    for dataset in ['lsun']:

        print(dataset)
        if dataset == 'icons':
            transform = transform_icons 
        root = datalocations[dataset]
        ood_dataset = ImageFolder(root, transform=transform, 
                                 target_transform=lambda x:-1)
        oodloader = torch.utils.data.DataLoader(ood_dataset, batch_size=64, 
                                         drop_last = True,
                                           shuffle = False, num_workers=4)
        
        auc,norm_auc = only_auc(testloader, oodloader,train_score_list, netG, netE, netD2, device, anom_metric_type='f_anomloss')
        aucs.append(auc)
        
        
    wandb.log({'lsun':auc, 'lsun_normauc': norm_auc}) #, 'isun': aucs[1],'imagenet':aucs[2], 'textures': aucs[3] })
    
#     with open(os.path.join(imgroot, "ood_aucs.txt"), "a") as f:
#                     f.write("{} \t {:.5f}\t {:.5f}\t {:.5f}\t {:.5f}".format(epoch, aucs[0], aucs[1], aucs[2], aucs[3]) + "\n")
                
    return aucs
        
  

        
        

        
    
