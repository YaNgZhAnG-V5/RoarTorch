import numpy as np
from torch.nn import functional as F
from captum.attr import DeepLiftShap


def compute_deep_sharp(model, preprocessed_image, label, baseline=None):
    if baseline == "zero":
        base_distribution = preprocessed_image.new_zeros((1,) + preprocessed_image.shape[1:])
    else:
        raise NotImplementedError
    saliency = DeepLiftShap(model).attribute(preprocessed_image, target=label, baselines=base_distribution)
    grad = saliency.detach().cpu().clone().numpy().squeeze()
    return grad
