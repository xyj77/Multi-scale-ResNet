import torch

def weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv3d') != -1:
        torch.nn.init.kaiming_uniform(m.weight.data)
        torch.nn.init.constant(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.kaiming_uniform(m.weight.data)
        torch.nn.init.constant(m.bias.data, 0.0) 