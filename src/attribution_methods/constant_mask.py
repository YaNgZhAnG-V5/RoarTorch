import numpy as np
import torch


def compute_constant_mask(model, preprocessed_image, label, baseline=None):
    grad = torch.ones_like(preprocessed_image).detach().cpu().clone().numpy().squeeze()
    return grad