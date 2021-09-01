import numpy as np
import torch
from skimage.draw import circle
from skimage import filters

def compute_constant_mask_circle(model, preprocessed_image, label, baseline=None):
    assert preprocessed_image.shape[-2] == preprocessed_image.shape[-1], "the image has different length and height"
    grad = torch.zeros_like(preprocessed_image).detach().cpu().clone().numpy()
    image_length = list(grad.shape)[-1]
    rr, cc = circle(image_length//2, image_length//2, image_length//6)
    grad[:, :, rr, cc] = 1
    grad = filters.gaussian(grad)
    return grad