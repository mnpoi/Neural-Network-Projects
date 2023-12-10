from torchvision.models import resnet152
from torch import nn

def get_resnet_model(num_classes, pretrained=True):
    net = resnet152(pretrained=pretrained)
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, num_classes)
    return net