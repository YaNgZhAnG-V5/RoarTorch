import numpy as np
from torch.nn import functional as F
from captum.attr import DeepLiftShap


def compute_deep_sharp(model, preprocessed_image, label, baseline=None):
    if baseline is "zero":
        base_distribution = preprocessed_image.new_zeros((10,) + preprocessed_image.shape[1:])
    else:
        raise NotImplementedError
    saliency = DeepLiftShap(model).attribute(preprocessed_image, label, baselines=base_distribution)
    grad = saliency.detach().cpu().clone().numpy().squeeze()
    return grad

attr_map = baseline.make_attribution(img, target, baselines=base_distribution)
