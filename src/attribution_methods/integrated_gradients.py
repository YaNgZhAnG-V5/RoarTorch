import numpy as np
from torch.nn import functional as F
from captum.attr import IntegratedGradients


def compute_integrated_gradients(model, preprocessed_image, label, baseline=None):
    saliency = IntegratedGradients(model).attribute(preprocessed_image, target=label, n_steps=35)
    grad = saliency.detach().cpu().clone().numpy().squeeze()
    return grad
