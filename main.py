import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import time
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.utils as utils
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils import data, tensorboard

from my_utils import *
from my_models import *
from config import *
from my_train import train

if __name__ == '__main__':
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device) # change allocation of current GPU
    today = datetime.date.today()
    print('오늘 날짜 :',today)
    print('cuda device :', device)
    
    # MNIST dataset 불러오기
    _, _, train_loader, test_loader = load_MNIST()
    
    pretrained_ResNet = torch.load('pretrained_model/best_ResNet.pt').to(device)
    enc = nn.Sequential(*(list(pretrained_ResNet.children())[:5])).to(device)
    gen = Generator().to(device)
    critic = Discriminator().to(device)
    initialize_weights(gen)
    initialize_weights(critic)

    # initializate optimizer
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
    opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

    # for tensorboard plotting
    fixed_noise = torch.randn(BATCH_SIZE, Z_DIM).to(device)
    writer_real = tensorboard.SummaryWriter(f"logs/GAN_MNIST/real2")
    writer_fake = tensorboard.SummaryWriter(f"logs/GAN_MNIST/fake2")
    writer_loss = tensorboard.SummaryWriter(f'logs/GAN_MNIST')
    
    if shape_test(enc, gen, critic, device):
        for param in enc.parameters():
            param.requires_grad = False
        
    step = 0
    for epoch in range(EPOCHS):
        loss_gen, loss_critic, img_grid_real, img_grid_fake = train(gen, critic, enc,
                                                                    train_loader, opt_gen, opt_critic,
                                                                    epoch, device)
        writer_real.add_image("Real", img_grid_real, global_step=step)
        writer_fake.add_image("Fake", img_grid_fake, global_step=step)
        writer_loss.add_scalars("Loss", {'gen':loss_gen,
                                         'critic':loss_critic,
                                         'total':loss_gen+loss_critic},
                                        global_step=step)
        step += 1 
