import torchvision.models as models

def VGG():
    return models.vgg16(num_classes=100)