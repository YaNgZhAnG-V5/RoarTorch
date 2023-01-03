import torch
from skimage.draw import circle
from skimage import filters


def compute_constant_circle_with_constant_class_mask(model, preprocessed_image, label, baseline=0.1, center_circle_size=0.7, size=0.2):
    # This method implements an attribution map that puts a circle in the center for all images.
    # Moreover, it puts additional class related constant circles for each class.
    # center_circle_size is the size of the center circle
    # Baselines is the margin between circle center and edge
    # margin insert mask to keep a margin to the edge of the image
    assert preprocessed_image.shape[-2] == preprocessed_image.shape[-1], "the image has different length and height"

    # create center circle
    grad = torch.zeros_like(preprocessed_image).detach().cpu().clone().numpy().squeeze()
    image_length = list(grad.shape)[-1]
    rr, cc = circle(image_length//2, image_length//2, int(image_length * 0.5 * center_circle_size))
    grad[:, rr, cc] = 1
    grad = filters.gaussian(grad)

    # get num of classes
    num_of_classes = model(preprocessed_image).shape[1]

    # get label
    label_idx = label.item()

    # construct circle based on label
    class_per_edge = torch.ceil(torch.tensor([num_of_classes/4.], dtype=torch.float64)).item()
    margin = int(image_length * baseline)
    stride = int((image_length - 2 * margin)/class_per_edge)
    edge_index = label_idx//class_per_edge
    position_index = label_idx % class_per_edge
    if edge_index == 0:
        center_x = margin
        center_y = margin
        center_x = center_x + stride * position_index
    elif edge_index == 1:
        center_x = image_length - margin
        center_y = margin
        center_y = center_y + stride * position_index
    elif edge_index == 2:
        center_x = image_length - margin
        center_y = image_length - margin
        center_x = center_x - stride * position_index
    elif edge_index == 3:
        center_x = margin
        center_y = image_length - margin
        center_y = center_y - stride * position_index
    rr, cc = circle((center_x, center_y), int(image_length * size), shape=(grad.shape[1], grad.shape[2]))
    grad[:, rr, cc] = 1

    # blur the mask
    grad = filters.gaussian(grad)
    return grad