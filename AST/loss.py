import torch.nn as nn

def criterion(outputs,labels):
    return nn.CrossEntropyLoss()(outputs,labels)