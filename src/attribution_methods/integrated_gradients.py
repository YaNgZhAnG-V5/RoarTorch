import numpy as np
from torch.nn import functional as F
from captum.attr import IntegratedGradients


def compute_integrated_gradients(model, preprocessed_image, label, baseline=None):
    self.attribute = IntegratedGradients(self.classifier).attribute
    saliency = DeepLiftShap(model).attribute(preprocessed_image, target=label)
    grad = saliency.detach().cpu().clone().numpy().squeeze()
    return grad
