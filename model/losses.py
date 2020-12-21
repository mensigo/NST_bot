import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        self.loss = F.mse_loss(self.target, self.target) # initializing

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def gram_matrix(input):
    _, f_map_num, h, w = input.size()  # batch size(=1)        
    features = input.view(f_map_num, h * w)
    G = torch.mm(features, features.t())
    return G.div(h * w * f_map_num)


class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
        self.loss = F.mse_loss(self.target, self.target) # initializing

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


def total_variation_loss(img):      
        diff_h = ((img[:, :, 1:, :] - img[:, :, :-1, :]) ** 2).sum()
        diff_w = ((img[:, :, :, 1:] - img[:, :, :, :-1]) ** 2).sum()    
        return diff_h + diff_w

