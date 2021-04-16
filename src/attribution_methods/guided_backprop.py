import numpy as np
from torch.nn import functional as F
from captum.attr import GuidedBackprop


def compute_guided_backprop(model, preprocessed_image, label, saliency_layer=None):
    saliency = GuidedBackprop(model).attribute(preprocessed_image, label)
    grad = saliency.detach().cpu().clone().numpy().squeeze()
    return grad
