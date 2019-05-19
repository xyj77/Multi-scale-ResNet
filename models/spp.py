import math
import torch
import torch.nn as nn

class SPPLayer(nn.Module):
    def __init__(self, scale_list):
        super(SPPLayer, self).__init__()
        self.scale_list = scale_list

    def forward(self, x):
        '''
        x: a tensor vector of previous convolution layer
        scale_list: list contain multi-scale pooling size
        
        returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
        '''    
        batch_size, in_channel, in_h, in_w = x.size()
        scale_list = self.scale_list
        for i in range(len(scale_list)):
            h_wid = int(math.ceil(in_h / scale_list[i]))
            w_wid = int(math.ceil(in_w / scale_list[i]))
            h_pad = (h_wid*scale_list[i] - in_h + 1)/2
            w_pad = (w_wid*scale_list[i] - in_w + 1)/2
            maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
            out = maxpool(x)
            if(i == 0):
                spp = out.view(batch_size, -1)
            else:
                spp = torch.cat((spp, out.view(batch_size, -1)), 1)
        return spp
        
class SPP3DLayer(nn.Module):
    def __init__(self, scale_list):
        super(SPP3DLayer, self).__init__()
        self.scale_list = scale_list

    def forward(self, x):
        '''
        x: a tensor vector of previous convolution layer
        scale_list: list contain multi-scale pooling size
        
        returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
        '''    
        batch_size, in_channel, in_h, in_w, in_t = x.size()
        scale_list = self.scale_list
        for i in range(len(scale_list)):
            h_wid = int(math.ceil(in_h / scale_list[i]))
            w_wid = int(math.ceil(in_w / scale_list[i]))
            t_wid = int(math.ceil(in_t / scale_list[i]))
            h_pad = (h_wid*scale_list[i] - in_h + 1)/2
            w_pad = (w_wid*scale_list[i] - in_w + 1)/2
            t_pad = (t_wid*scale_list[i] - in_t + 1)/2
            maxpool = nn.MaxPool3d((h_wid, w_wid, t_wid), stride=(h_wid, w_wid, t_wid), padding=(h_pad, w_pad, t_pad))
            out = maxpool(x)
            if(i == 0):
                spp = out.view(batch_size, -1)
            else:
                spp = torch.cat((spp, out.view(batch_size, -1)), 1)
        return spp