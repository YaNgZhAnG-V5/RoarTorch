import numpy as np
from torch.nn import functional as F
from captum.attr import GuidedBackprop


def compute_guided_backprop(model, preprocessed_image, label):
    saliency = GuidedBackprop(model).attribute(preprocessed_image, label)
    image_shape = (preprocessed_image.shape[-2], preprocessed_image.shape[-1])
    saliency = F.interpolate(saliency, image_shape, mode="bilinear", align_corners=False)
    grad = saliency.detach().cpu().clone().numpy()  # 1, 1, 8, 8 for cifar10_resnet8
    grad = np.concatenate((grad,) * 3, axis=1).squeeze()  # 3, 8, 8
    return grad
