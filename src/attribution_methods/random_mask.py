import numpy as np
import torch


def compute_random_mask(model, preprocessed_image, label, baseline=None):
    grad = torch.rand_like(preprocessed_image).detach().cpu().clone().numpy().squeeze()
    return grad