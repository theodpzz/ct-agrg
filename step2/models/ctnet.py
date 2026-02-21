import torch
import torch.nn as nn
from torchvision import models
    
##########################
class CTNet(nn.Module):
    """
    Pretrain.
    """
    def __init__(self, args):
        super(CTNet, self).__init__()  

        # args
        self.args = args
        n_outputs = self.args.n_outputs

        # resnet modules
        resnet = models.resnet18(weights=None)
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

    def load(self):
        self.load_state_dict(torch.load(self.args.path_ctnet, map_location=self.args.device))

    def adjust(self):
        self.classifier  = nn.Sequential(*list(self.classifier.children())[:3])
        self.classifier_ = nn.Sequential(*(list(self.classifier.children())[3:]))

    def forward(self, volume):

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

        # classifier head
        x = x.view(batch_size, 16*18*5*5)
        
        x = self.classifier(x)

        return x
    

class CTNetStep(nn.Module):
    """
    Step 1 and 2.
    """
    def __init__(self, args):
        super(CTNetStep, self).__init__()  

        self.args = args

        # pre-trained backbone
        self.encoder = CTNet(args)
        self.encoder.adjust()

        # classification heads : one per label
        self.heads_in  = nn.Sequential(nn.Linear(2048, 1024*18), nn.ReLU(True), nn.Dropout(0.5))
        self.heads_out = nn.ModuleList([nn.Sequential(nn.Linear(1024, 128), nn.ReLU(True), nn.Dropout(0.5), nn.Linear(128, 1)) for _ in range(18)])

        # loss function
        self.loss = nn.BCEWithLogitsLoss(reduction = 'none')

        # sigmoid to compute probabilities
        self.sigmoid = nn.Sigmoid()

    def load(self):
        self.load_state_dict(torch.load(self.args.path_ctnetstep, map_location=self.args.device))

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def getloss(self, prediction, target):
        # compute BCE Loss
        loss = self.loss(prediction, target)
        return loss  

    def extract_features(
        self,
        volume: torch.tensor,
    ):
        # extract embeddings from pre-trained backbone
        embeddings = self.encoder(volume)

        # projection from 2048 feature space to 1024 feature space
        hidden_states = self.heads_in(embeddings)

        # reshape to split as multiple embeddings
        hidden_states = hidden_states.view(-1, 18, 1024)

        # projection from 1024 feature space to 1, and reshape logits as (1, num_labels)
        logits = torch.cat([head(hidden_states[:, i]) for i, head in enumerate(self.heads_out)], dim=1)

        # logits to probablities
        predictions = self.sigmoid(logits)

        return hidden_states, predictions
    
    def forward(self, volume, labels, return_embeddings=False):

        # extract embeddings from pre-trained backbone
        embeddings = self.encoder(volume)

        # projection from 2048 feature space to 1024 feature space
        hidden_states = self.heads_in(embeddings)

        # reshape to split as multiple embeddings
        hidden_states = hidden_states.view(-1, 18, 1024)

        # projection from 1024 feature space to 1, and reshape logits as (1, num_labels)
        logits = torch.cat([head(hidden_states[:, i]) for i, head in enumerate(self.heads_out)], dim=1)

        # compute loss
        loss = self.getloss(logits, labels)

        # logits to probablities
        predictions = self.sigmoid(logits)

        if return_embeddings:
            return hidden_states, predictions, loss.mean(dim=0).unsqueeze(0)
        else:
            return predictions, loss.mean(dim=0).unsqueeze(0)
