import os
import time
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.nn import Conv2d, ReLU, AvgPool2d, Sequential
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models

from model.image_loader import *
from model.losses import *


# basic constaints
IMG_SIZE = 256
ITER_NUM = 100 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
STYLE_WEIGHT = 1e5
TV_WEIGHT = 5e-6
TV_STYLE_COEFF = 5e-12
VGG19_W_PATH = 'model/vgg19_conv1_1-relu4_2.pt'
LAYER_NAMES = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
               'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
               'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2',
               'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
               'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2']
CONTENT_LAYERS = ['relu4_2']
STYLE_LAYERS = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_2']
STYLE_LOSS_WEIGHTS = np.array([0.5,.4,.3,.2]) / 1.4
VGG_MEAN = torch.Tensor([0.485, 0.456, 0.406])
VGG_STD = torch.Tensor([0.229, 0.224, 0.225])

class Model_vgg19_cut(nn.Module):

    def __init__(self):

        super().__init__() 
        self.conv1_1 = Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu1_1 = ReLU()
        self.conv1_2 = Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu1_2 = ReLU()
        self.pool1 = AvgPool2d(kernel_size=2, stride=2)

        self.conv2_1 = Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2_1 = ReLU()
        self.conv2_2 = Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu2_2 = ReLU()
        self.pool2 = AvgPool2d(kernel_size=2, stride=2)

        self.conv3_1 = Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu3_1 = ReLU()
        self.conv3_2 = Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu3_2 = ReLU()
        self.conv3_3 = Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu3_3 = ReLU()
        self.conv3_4 = Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu3_4 = ReLU()
        self.pool3 = AvgPool2d(kernel_size=2, stride=2)

        self.conv4_1 = Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu4_1 = ReLU()
        self.conv4_2 = Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu4_2 = ReLU()
                      
    def forward(self,x):
        
        block1 = Sequential(self.conv1_1, self.relu1_1,
                            self.conv1_2, self.relu1_2, self.pool1)(x)
        block2 = Sequential(self.conv2_1, self.relu2_1,
                            self.conv2_2, self.relu2_2, self.pool2)(block1)
        block3 = Sequential(self.conv3_1, self.relu3_1,
                            self.conv3_2, self.relu3_2,
                            self.conv3_3, self.relu3_3,
                            self.conv3_4, self.relu3_4, self.pool3)(block2)
        block4 = Sequential(self.conv4_1, self.relu4_1,
                            self.conv4_2, self.relu4_2)(block3)
        return block4


def get_model_and_losses(image_loader,
                         style_num,
                         content_layers,
                         style_layers):
    
    cnn = Model_vgg19_cut()
    cnn.load_state_dict(torch.load(VGG19_W_PATH, map_location=DEVICE))
    model = nn.Sequential()

    content_losses = []
    style_losses = []
    for i, layer in enumerate(cnn.children()):
        
        name = LAYER_NAMES[i]            
        model.add_module(name, layer.to(DEVICE))

        # add content loss
        if name in content_layers:
            
            target = model(image_loader.get_content()).detach()
            content_loss = ContentLoss(target)
            model.add_module('content_loss_{}'.format(name[-3]), content_loss)
            content_losses.append(content_loss)

        # add style loss
        if name in style_layers:
            
            style_img = image_loader.get_style(style_num)
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature) 
            model.add_module('style_loss_{}'.format(name[-3]), style_loss)
            style_losses.append(style_loss)

    del cnn
    model.eval()
    return model, style_losses, content_losses


def run_style_transfer(image_loader,
                       input_img,
                       style_num,
                       num_steps,
                       content_layers,
                       style_layers,
                       style_loss_weights,
                       style_weight,
                       content_weight,
                       tv_weight,
                       print_flg):
    
    if (print_flg):
        start_time = time.time()
        print('Building the style transfer model..')

    model, style_losses, content_losses = \
        get_model_and_losses(
            image_loader=image_loader,
            style_num=style_num,
            content_layers=content_layers,
            style_layers=style_layers)
            
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    
    if (print_flg):
    	print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():

            optimizer.zero_grad()
            model(input_img) # forward pass

            # collect losses from modules
            style_score = 0
            content_score = 0
            for j,sl in enumerate(style_losses):
                style_score += sl.loss * style_loss_weights[j]
            for cl in content_losses:
                content_score += cl.loss
            
            # style-content weighting
            style_score *= style_weight
            content_score *= content_weight
            tv_score =  tv_weight * total_variation_loss(input_img)

            loss = style_score + content_score + tv_score
            loss.backward()
            
            # progress
            run[0] += 1
            if (run[0] % 100 == 0) and (print_flg):
                
                print('Iter {}:'.format(run[0]))
                print('Style Loss : {:.4f} Content Loss: {:.4f} TV Loss: {:.4f}'\
                      .format(style_score.item(), content_score.item(), tv_score.item()))
                print('{} sec'.format(int(time.time()-start_time)))
                print()

            return style_score + content_score + tv_score
        
        optimizer.step(closure)

    if (print_flg):
    	print('Style is transferred in {} seconds.'.format(int(time.time()-start_time)))

    return input_img


async def apply_NST(content_path, style_path, save_path, style_weight):

    image_loader = Image_loader([content_path], [style_path],
                                IMG_SIZE, DEVICE, VGG_MEAN, VGG_STD, True)
    input_img = image_loader.get_content()

    output_img = run_style_transfer(image_loader=image_loader,
                                    input_img=input_img,
                                    style_num=0,
                                    num_steps=ITER_NUM,
                                    content_layers=CONTENT_LAYERS,
                                    style_layers=STYLE_LAYERS,
                                    style_loss_weights=STYLE_LOSS_WEIGHTS,
                                    style_weight=style_weight,
                                    content_weight=1,
                                    tv_weight=TV_WEIGHT,
                                    print_flg=True)

	tensor2PIL(output_img, VGG_MEAN, VGG_STD).save(save_path)


if __name__ == '__main__':

    from image_loader import *
    from losses import *

    test_content_path = '../test_content.jpg'
    test_style_path = '../test_style.jpg'
    test_output_path = '../test_result.jpg'
    image_loader = Image_loader([test_content_path], [test_style_path],
                                IMG_SIZE, DEVICE, VGG_MEAN, VGG_STD, True)
    input_img = image_loader.get_content().clone()
    
    output_img = run_style_transfer(image_loader=image_loader,
                                    input_img=input_img,
                                    style_num=0,
                                    num_steps=ITER_NUM,
                                    content_layers=CONTENT_LAYERS,
                                    style_layers=STYLE_LAYERS,
                                    style_loss_weights=STYLE_LOSS_WEIGHTS,
                                    style_weight=STYLE_WEIGHT,
                                    content_weight=1,
                                    tv_weight=STYLE_WEIGHT*TV_STYLE_COEFF,
                                    print_flg=True)

    tensor2PIL(output_img, VGG_MEAN, VGG_STD).save(test_output_path)

