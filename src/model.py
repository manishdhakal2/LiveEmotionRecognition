import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class EmotionRecognizerResnet(nn.Module):
    def __init__(self,image_size, class_num):

        self.class_num = class_num
        self.image_size = image_size
        
        self.resnet = resnet50(weights = ResNet50_Weights.DEFAULT)


        for params in self.resnet.parameters():
            params.requires_grad = False

        self.resnet.fc = nn.Linear(self.resnet.fc.in_features,self.class_num)

    def forward(self,x):
        return self.resnet (x)
    


