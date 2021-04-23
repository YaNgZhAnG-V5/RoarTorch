import torch
import torchvision.models as models

def VGG():
    return models.vgg16(num_classes=10)

def VGG_cars():
    model = models.vgg16(pretrained=True)
    model.classifier[6] = torch.nn.Linear(4096, 196)
    return model