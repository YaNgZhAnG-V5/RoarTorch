import numpy as np
from torch.nn import functional as F
from torchray.attribution.extremal_perturbation import extremal_perturbation


def compute_extremal_perturbation(model, preprocessed_image, label, saliency_layer=None):
    saliency, _ = extremal_perturbation(model, preprocessed_image, label.item())
    grad = saliency.detach().cpu().clone().numpy()
    grad = np.concatenate((grad,) * 3, axis=1).squeeze()
    return grad
