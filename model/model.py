import os
import time
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models

from model.image_loader import *
from model.losses import *


# basic constaints
IMG_SIZE = 300
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ITER_NUM = 500
STYLE_WEIGHT = 1e7


vgg19_names_list = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
                   'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
                   'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
                    'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
                   'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
                    'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
                   'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
                    'relu5_3', 'conv5_4', 'relu5_4', 'pool5']


content_layers_default = ['conv4_2']
style_layers_default = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']


class Normalization(nn.Module):

    def __init__(self):
        super(Normalization, self).__init__()
        # imagenet stats
        vgg_norm_mean = torch.tensor([0.485, 0.456, 0.406])
        vgg_norm_std = torch.tensor([0.229, 0.224, 0.225])
        self.mean = vgg_norm_mean.view(-1, 1, 1).to(DEVICE)
        self.std = vgg_norm_std.view(-1, 1, 1).to(DEVICE)

    def forward(self, img):
        return (img - self.mean) / self.std


def get_model_and_losses(image_loader,
	                     style_num,
	                     content_layers,
	                     style_layers):
    
    cnn = models.vgg19(pretrained=True).features
    normalization = Normalization().to(DEVICE)
    model = nn.Sequential(normalization) # new model

    content_losses = []
    style_losses = []
    for i, layer in enumerate(cnn.children()):
        
        name = vgg19_names_list[i]
        if (name == 'relu5_1'):
            break # trim off excess layers
            
        # add layers            
        if isinstance(layer, nn.Conv2d):
            out_c = layer.out_channels
            instance_norm = nn.InstanceNorm2d(num_features=out_c) # add instNorm
            model.add_module(name, instance_norm.to(DEVICE))  
        elif isinstance(layer, nn.ReLU):
            layer = nn.ReLU(inplace=False) # turn off inplace
        elif isinstance(layer, nn.MaxPool2d):
            layer = nn.AvgPool2d(kernel_size=2, stride=2) # replace maxp w/ avgp
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
        model.add_module(name, layer.to(DEVICE))

        # add content loss
        if name in content_layers:

            content_img = image_loader.get_content()
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(name[-3]), content_loss)
            content_losses.append(content_loss)

        # add style loss
        if name in style_layers:
            
            style_img = image_loader.get_style(style_num)
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)      
            model.add_module("style_loss_{}".format(name[-3]), style_loss)
            style_losses.append(style_loss)

    model = model.eval()
    return model, style_losses, content_losses


def get_input_optimizer(input_img):
    # input requires a gradient
    optimizer = optim.LBFGS([input_img.requires_grad_()]) 
    return optimizer


def run_style_transfer(image_loader,
                       input_img,
                       style_num,
                       num_steps,
                       style_loss_weights,
                       style_weight,
                       content_weight,
                       print_flg):
    
    if (print_flg):
    	start_time = time.time()
    	print('Building the style transfer model..')
    model, style_losses, content_losses = \
        get_model_and_losses(
            image_loader=image_loader,
            style_num=style_num,
            content_layers=content_layers_default,
            style_layers=style_layers_default)
            
    optimizer = get_input_optimizer(input_img)
    
    if (print_flg):
    	print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():

            input_img.data.clamp_(0, 1)  # correct the values 

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

            loss = style_score + content_score
            loss.backward()
            
            # progress
            run[0] += 1
            if (run[0] % 100 == 0) and (print_flg):
                
                print("Iter {}:".format(run[0]))
                print('Style Loss : {:4f} Content Loss : {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score
        
        optimizer.step(closure)

    if (print_flg):
    	print('Style is transferred in {} seconds.'.format(int(time.time()-start_time)))

    # last correction
    input_img.data.clamp_(0, 1)
    return input_img


async def apply_NST(content_path, style_path, save_path, style_weight):

	image_loader = Image_loader([content_path], [style_path], IMG_SIZE, DEVICE, True)

	input_img = image_loader.get_content().clone()
	# input_img = torch.randn(image_loader.get_content().data.size(), device=DEVICE)

	output_img = run_style_transfer(image_loader=image_loader,
		                            input_img=input_img,
	                                style_num=0,
		                            num_steps=ITER_NUM,
	                                style_loss_weights=[1/len(style_layers_default)] *
	                                                    len(style_layers_default),
		                            style_weight=style_weight,
		                            content_weight=1,
		                            print_flg=True)

	tensor2PIL(output_img).save(save_path)


if __name__ == '__main__':

    from image_loader import *
    from losses import *

    test_content_path = '../test_content.jpg'
    test_style_path = '../test_style.jpg'
    test_output_path = '../test_result.jpg'
    image_loader = Image_loader([test_content_path], [test_style_path],
    							IMG_SIZE, DEVICE, True)
    input_img = image_loader.get_content().clone()
    # start from noise
    # input_img = torch.randn(image_loader.get_content().data.size(), device=DEVICE)
    
    output_img = run_style_transfer(image_loader=image_loader,
                                    input_img=input_img,
                                    style_num=0,
                                    num_steps=ITER_NUM,
                                    style_loss_weights=[1/len(style_layers_default)] *
                                                        len(style_layers_default),
                                    style_weight=STYLE_WEIGHT,
                                    content_weight=1,
                                    print_flg=True)

    tensor2PIL(output_img).save(test_output_path)

