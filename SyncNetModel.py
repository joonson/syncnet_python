import torch
import torch.nn as nn

cuda = False

def save(model, filename):
    with open(filename, "wb") as f:
        torch.save(model, f);
        print("%s saved."%filename);

def load(filename):
    net = torch.load(filename)
    return net;
    
class S(nn.Module):
    def __init__(self, num_layers_in_fc_layers = 1024):
        super(S, self).__init__();

        self.__nFeatures__ = 24;
        self.__nChs__ = 32;
        self.__midChs__ = 32;

        self.netcnnaud = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1,1), stride=(1,1)),

            nn.Conv2d(64, 192, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,3), stride=(1,2)),

            nn.Conv2d(192, 384, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,3), stride=(2,2)),
            
            nn.Conv2d(256, 2048, kernel_size=(5,4), padding=(0,0)),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
        );

        self.netfcaud = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, num_layers_in_fc_layers),
        );

        self.netfclip = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, num_layers_in_fc_layers),
        );

        self.netcnnlip = nn.Sequential(
            nn.Conv3d(3, 96, kernel_size=(5,7,7), stride=(1,2,2), padding=0),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2)),

            nn.Conv3d(96, 256, kernel_size=(1,5,5), stride=(1,2,2), padding=(0,1,1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),

            nn.Conv3d(256, 256, kernel_size=(1,3,3), padding=(0,1,1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),

            nn.Conv3d(256, 256, kernel_size=(1,3,3), padding=(0,1,1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),

            nn.Conv3d(256, 256, kernel_size=(1,3,3), padding=(0,1,1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2)),

            nn.Conv3d(256, 2048, kernel_size=(1,6,6), padding=0),
            nn.BatchNorm3d(2048),
            nn.ReLU(inplace=True),
        );
        
        if cuda:
            self.netcnnaud = self.netcnnaud.cuda();
            self.netcnnlip = self.netcnnlip.cuda();
            self.netfcaud  = self.netfcaud.cuda();
            self.netfclip = self.netfclip.cuda();

    def forward_aud(self, x):

        mid = self.netcnnaud(x); # N x ch x 24 x M
        mid = mid.view((mid.size()[0], -1)); # N x (ch x 24)
        out = self.netfcaud(mid);

        return out;

    def forward_lip(self, x):

        mid = self.netcnnlip(x); 
        mid = mid.view((mid.size()[0], -1)); # N x (ch x 24)
        out = self.netfclip(mid);

        return out;
