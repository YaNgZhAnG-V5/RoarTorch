import numpy as np
import torch
from skimage.draw import circle
from skimage import filters


def compute_constant_class_mask(model, preprocessed_image, label, baseline=None):
    assert preprocessed_image.shape[-2] == preprocessed_image.shape[-1], "the image has different length and height"

    # get num of classes
    num_of_classes = model(preprocessed_image).shape[1]
    classes_per_row = torch.floor(torch.sqrt(torch.tensor([num_of_classes]))).item()
    print(classes_per_row)

    # get label
    label_idx = label.item()

    # get image length
    image_length = list(grad.shape)[-1]

    # construct circle based on label
    grad = torch.zeros_like(preprocessed_image).detach().cpu().clone().numpy().squeeze()
    center_x = image_length * ((label_idx%classes_per_row)+1)/(classes_per_row+1)
    center_y = image_length * ((label_idx//classes_per_row)+1)/(classes_per_row+1)
    rr, cc = circle(center_x, center_y, image_length // (classes_per_row*2))
    grad[:, rr, cc] = 1
    grad = filters.gaussian(grad)
    return grad
