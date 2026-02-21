import torch
import torch.nn as nn

from torchvision import models

class CTNet(nn.Module):
    """
    Model for visual feature extraction.
    """
    def __init__(self, args):
        super(CTNet, self).__init__()  

        self.args = args
        n_outputs = args['parameters']['n_outputs']

        # resnet
        resnet = models.resnet18(weights=None)
        resnet.load_state_dict(torch.load(args['model']['path_resnet']))
        self.features = nn.Sequential(*(list(resnet.children())[:-2]))
        
        # 3D convolution to reduce feature space
        self.reducingconvs = nn.Sequential(
            nn.Conv3d(80, 64, kernel_size = (3,3,3), stride=(3,1,1), padding=0),
            nn.ReLU(),
            
            nn.Conv3d(64, 32, kernel_size = (3,3,3), stride=(3,1,1), padding=0),
            nn.ReLU(),
            
            nn.Conv3d(32, 16, kernel_size = (3,2,2), stride=(3,2,2), padding=0),
            nn.ReLU())
        
        # classifier head
        self.classifier = nn.Sequential(
            nn.Linear(16*18*5*5, 2048),
            nn.ReLU(True),
            nn.Dropout(0.5),
            
            nn.Linear(2048, 512), 
            nn.ReLU(True),
            nn.Dropout(0.5),

            nn.Linear(512, 128), 
            nn.ReLU(True),
            nn.Dropout(0.5),
            
            nn.Linear(128, n_outputs))
        
        # loss function
        self.loss = nn.BCEWithLogitsLoss(reduction = 'none')

        # sigmoid to compute probabilities
        self.sigmoid = nn.Sigmoid()

    def getloss(self, prediction, target):
        loss = self.loss(prediction, target)
        return loss  

    def forward(self, volume, labels):

        # shape of input as a list
        shape = list(volume.size())
        # extract batch size
        batch_size = int(shape[0])
        
        # extract embeddings with resnet
        x = volume.view(batch_size*80, 3, 480, 480)
        x = self.features(x)

        # 3D convolutions to reduce dimensions
        x = x.view(batch_size, 80, 512, 15, 15)
        x = self.reducingconvs(x)

        x = x.view(batch_size, 16*18*5*5)
        
        # classification head
        logits = self.classifier(x)
    
        loss = self.getloss(logits, labels.squeeze(1))
    
        return loss

    def predict(self, volume):

        # shape of input as a list
        shape = list(volume.size())
        # extract batch size
        batch_size = int(shape[0])
        
        # extract embeddings with resnet
        x = volume.view(batch_size*80, 3, 480, 480)
        x = self.features(x)

        # 3D convolutions to reduce dimensions
        x = x.view(batch_size, 80, 512, 15, 15)
        x = self.reducingconvs(x)

        x = x.view(batch_size, 16*18*5*5)
        
        # classification head
        logits = self.classifier(x)
    
        # logits to probablities
        predictions = self.sigmoid(logits)
    
        return predictions