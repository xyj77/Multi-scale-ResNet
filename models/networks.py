import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingNet2D(nn.Module):
    def __init__(self, in_channel, out_length):
        super(EmbeddingNet2D, self).__init__()
        self.out_length = out_length
        self.convnet = nn.Sequential(nn.Conv2d(in_channel, 6, 3), nn.PReLU(), #inx32x32->6x30x30
                                     nn.MaxPool2d(2, stride=2),       #6x30x30->6x15x15
                                     nn.Conv2d(6, 8, 4), nn.PReLU(),  #6x15x15->8x12x12
                                     nn.MaxPool2d(2, stride=2),       #8x12x12->8x6x6
                                     # nn.AvgPool2d(6)
                                     )      

        self.fc = nn.Sequential(nn.Linear(8*6*6, 32),
                                nn.PReLU(),
                                nn.Linear(100, out_length)
                                )
    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)
        
class EmbeddingNet3D(nn.Module):
    def __init__(self, in_channel, out_length):
        super(EmbeddingNet3D, self).__init__()
        self.out_length = out_length
        self.convnet = nn.Sequential(nn.Conv3d(in_channel, 16, (3,3,3)),#inx5x32x32->6x3x30x30
                                     nn.PReLU(),
                                     # nn.Dropout3d(p=0.5, inplace=False),
                                     nn.MaxPool3d((1,2,2)),                 #6x3x30x30->6x3x15x15
                                     nn.Conv3d(16, 32, (3,4,4)),            #6x3x15x15->8x1x12x12
                                     nn.PReLU(), 
                                     # nn.Dropout3d(p=0.5, inplace=False),
                                     nn.MaxPool3d((1,2,2)),                 #8x1x12x12->8x1x6x6
                                     # nn.Dropout3d(p=0.5, inplace=False),
                                     nn.Conv3d(32, 32, (1,3,3)),            #8x1x6x6->8x1x4x4
                                     nn.PReLU(),
                                     # nn.AvgPool3d((1,4,4))
                                     )      

        self.fc = nn.Sequential(nn.Linear(32*1*4*4, 32),
                                nn.PReLU(),
                                nn.Dropout(p=0.5, inplace=False),
                                nn.Linear(32, out_length),
                                nn.PReLU(),
                                nn.Dropout(p=0.5, inplace=False)
                                )
    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)      
       
      
class EmbeddingNet2D40(nn.Module):
    def __init__(self, in_channel, out_length):
        super(EmbeddingNet2D40, self).__init__()
        self.out_length = out_length
        self.convnet = nn.Sequential(nn.Conv2d(in_channel, 8, 3), nn.PReLU(), #inx40x40->8x38x38
                                     nn.MaxPool2d(2, stride=2),       #8x38x38->8x19x19
                                     nn.Conv2d(8, 16, 4), nn.PReLU(), #8x19x19->16x16x16
                                     nn.MaxPool2d(2, stride=2),       #16x16x16->16x8x8
                                     nn.Conv2d(16, 32, 3), nn.PReLU(),#16x8x8->32x6x6
                                     nn.AvgPool2d(6)               #32x6x6->32x1x1
                                     )      

        self.fc = nn.Sequential(nn.Linear(32, 32),
                                nn.PReLU(),
                                nn.Linear(32, out_length)
                                )
    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)
        
class EmbeddingNet3D40(nn.Module):
    def __init__(self, in_channel, out_length):
        super(EmbeddingNet3D40, self).__init__()
        self.out_length = out_length
        self.convnet = nn.Sequential(nn.Conv3d(in_channel, 8, (3,3,3), padding=(1,0,0)),#inx5x40x40->8x5x38x38
                                     nn.PReLU(), 
                                     # nn.Dropout3d(p=0.5, inplace=False),
                                     nn.MaxPool3d((1,2,2)),                 #8x5x38x38->8x5x19x19
                                     nn.Conv3d(8, 16, (3,4,4)), nn.PReLU(), #8x5x19x19->16x3x16x16
                                     # nn.Dropout3d(p=0.5, inplace=False),
                                     nn.MaxPool3d((1,2,2)),                 #16x3x16x16->16x3x8x8
                                     nn.Conv3d(16, 32, (3,3,3)), nn.PReLU(),#16x3x8x8->32x1x6x6
                                     # nn.Dropout3d(p=0.5, inplace=False),
                                     nn.AvgPool3d((1,6,6))                  #32x6x6->32x1x1
                                     )      

        self.fc = nn.Sequential(nn.Linear(32, 32),
                                nn.PReLU(),
                                nn.Dropout(p=0.5, inplace=False),
                                nn.Linear(32, out_length),
                                nn.Dropout(p=0.5, inplace=False)
                                )
    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)

class ClassificationNet(nn.Module):
    def __init__(self, embedding_net, n_classes):
        super(ClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nolinear = nn.PReLU()
        self.fc1 = nn.Linear(embedding_net.out_length, n_classes)

    def forward(self, x):
        output = self.embedding_net(x)
        output = self.nolinear(output)
        scores = F.log_softmax(self.fc1(output), dim=-1)
        return scores

    def get_embedding(self, x):
        return self.nolinear(self.embedding_net(x))

class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)

class CompareNet(nn.Module):
    def __init__(self, embedding_net, feature_num):
        super(CompareNet, self).__init__()
        self.embedding_net = embedding_net
        self.fc = nn.Sequential(nn.Linear(feature_num, 32),
                                nn.PReLU(),
                                nn.Dropout(p=0.5, inplace=False),
                                nn.Linear(32, 2),
                                nn.PReLU()
                                )
        
    def get_embedding(self, x):
        return self.embedding_net(x)

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        concat = torch.cat((output1,output2), 1)
        scores = F.log_softmax(self.fc(concat), dim=-1)
        return scores