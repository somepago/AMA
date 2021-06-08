from __future__ import print_function
import os
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
import torch.nn.functional as F



def calc_gradient_penalty(netD2, real_data, fake_data, Lambda = 10 ):
    device = real_data.device
    b_size = real_data.shape[0]
    alpha = torch.unsqueeze(torch.unsqueeze((torch.rand(b_size, 1)), -1), -1)
    # alpha = torch.rand(b_size, 1)
    alpha = alpha.to(device)
    #this finds stuff on the line between real and fake 
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    # interpolates = real_data
    interpolates = interpolates.to(device)
    interpolates = torch.tensor(interpolates, requires_grad = True)
#     interpolates = interpolates.clone().detach().requires_grad_(True)
    real_inter = torch.cat((real_data,interpolates),dim=1)
    #Runs the discriminator on the resulting interpolated points
    disc_interpolates = netD2(real_inter)

    #Calculates the gradient
    gradients = autograd.grad(outputs = disc_interpolates, inputs=real_inter, grad_outputs = torch.ones(disc_interpolates.size()).to(device) , create_graph = True, retain_graph = True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * Lambda
    return gradient_penalty

def recon_discriminator_loss(images,labels, netD2, netE, netG ,losstype='wasserstein',anom_lambda=1, use_penalty = True, gradp_lambda = 10, ip_lambda = 0.5):
    device = labels.device
    real_idx = ((labels > 0).nonzero()).squeeze(1)
    labels = torch.tensor([c if c==1 else anom_lambda*c for c in labels]).to(device)
    rec_real = netG( netE(images)).detach()
    
    x1 = images[real_idx]
    real_labels = labels[real_idx]
    alpha1 = torch.FloatTensor(x1.size(0), 1).uniform_(0, 0.5).to(device)
    alpha2 = torch.FloatTensor(x1.size(0), 1).uniform_(0, 0.5).to(device)
    e1 = netE(x1).detach()
    idx1 = torch.randperm(x1.size(0))  
    idx2 = torch.randperm(x1.size(0))
    x2 = x1[idx1]
    e2 = e1[idx1]
    x3 = x1[idx2]
    e3 = e1[idx2]
    interpolates = alpha1*e1 + alpha2*e2 + (1 - alpha1 - alpha2 )*e3
    ri = netG(interpolates).detach()

    interpolation_loss = (wass_loss(x1,ri,real_labels, netG, netE,netD2) + wass_loss(x2,ri,real_labels, netG, netE,netD2) + wass_loss(x3,ri,real_labels, netG, netE,netD2) )/3
    
    disc_loss = wass_loss(images,rec_real,labels, netG, netE,netD2)
#     z = torch.zeros([images.shape[0], 128], dtype=torch.float32, device=device )
#     awayloss = wass_loss(images,netG(z).detach(),labels, netG, netE,netD2)

    if(use_penalty):
        fake = netG(netE(images)).detach()
        lossD2 = calc_gradient_penalty(netD2, images, fake, Lambda = gradp_lambda)
        disc_loss+= lossD2

    return disc_loss + ip_lambda* (interpolation_loss )

def recon_loss_joint(images, labels,netG, netE, netD2, loss_type = 'wasserstein',anom_lambda=1, ip_lambda = 0.5):
    rec_real = netG(netE(images))
    real_rec = torch.cat((images,rec_real),dim=1)
    device = labels.device
    real_idx = ((labels > 0).nonzero()).squeeze(1)
    labels = torch.tensor([c if c==1 else anom_lambda*c for c in labels]).to(device)
    x1 = images[real_idx]
    real_labels = labels[real_idx]
    alpha1 = torch.FloatTensor(x1.size(0), 1).uniform_(0, 0.5).to(device)
    alpha2 = torch.FloatTensor(x1.size(0), 1).uniform_(0, 0.5).to(device)
    e1 = netE(x1).detach()
    idx1 = torch.randperm(x1.size(0))  
    idx2 = torch.randperm(x1.size(0))
    x2 = x1[idx1]
    e2 = e1[idx1]
    x3 = x1[idx2]
    e3 = e1[idx2]
    interpolates = alpha1*e1 + alpha2*e2 + (1 - alpha1 - alpha2 )*e3
    ri = netG(interpolates).detach()
    interpolation_loss = (-(netD2(torch.cat((x1,ri),dim =1)).view(-1)*real_labels.float()).mean() -(netD2(torch.cat((x2,ri),dim =1)).view(-1)*real_labels.float()).mean()-(netD2(torch.cat((x3,ri),dim =1)).view(-1)*real_labels.float()).mean() )/3
#     labels = torch.tensor([c if c==1 else anom_lambda*c for c in labels]).to(device)
    if loss_type == 'wasserstein':
        recon_loss_encoded = -(netD2(real_rec).view(-1)*labels.float()).mean()
    elif loss_type == 'hinge':
        recon_loss_encoded = nn.ReLU()(1.0 + (labels.float())*(netD2(real_rec).view(-1))).mean()
    elif loss_type == 'l1':
        recon_loss_encoded = -torch.mean(torch.mean(torch.abs(images - rec_real),dim=(1,2,3))*labels.float())
    else:
        print(f'unknown reconstruction loss type in recon_loss_joint fucntion!: {losstype}')
        raise
#     z = torch.zeros([images.shape[0], 128], dtype=torch.float32, device=device )
#     awayloss = wass_loss(images,netG(z).detach(),labels, netG, netE,netD2)
    return recon_loss_encoded + ip_lambda* (interpolation_loss) 


def wass_loss(images,interpolates,labels, netG, netE,netD2):
    real_real = torch.cat((images,images),dim =1)
    real_recreal = torch.cat((images,interpolates),dim =1)
    
    disc_loss = -(netD2(real_real).view(-1)*labels.float()).mean()  + (netD2(real_recreal).view(-1)*labels.float()).mean()
    return disc_loss