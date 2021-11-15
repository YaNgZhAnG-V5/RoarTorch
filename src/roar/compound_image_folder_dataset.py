from copy import deepcopy

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader

from src.dataset import factory as dataset_factory
from src.roar import roar_core
from src.utils.sysutils import get_cores_count


class CompoundImageFolderDataset(torch.utils.data.Dataset):
    """
    To load Image Folder dataset along with images attribution files.
    """

    def __init__(self, dataset_name,
                 image_files_train_path,
                 image_files_validation_path,
                 image_files_test_path,
                 attribution_files_train_path,
                 attribution_files_validation_path,
                 attribution_files_test_path,
                 save_image=True,
                 num_saved_images=10,
                 percentile=0.1,
                 roar=True,
                 non_perturbed_testset=True):
        """

        Args:
            dataset_name:
            image_files_train_path:
            image_files_validation_path:
            image_files_test_path:
            attribution_files_train_path:
            attribution_files_validation_path:
            attribution_files_test_path:
            save_image: whether to save images (original & perturbed & augmented) for debugging
            num_saved_images: number of saved images
            percentile: The % of pixels to remove from input image.
            roar: Set to True for ROAR metric, False for KAR metric.
            non_perturbed_testset: Set to True if we don't want to perturb testset images
        """

        if dataset_name not in dataset_factory.MAP_DATASET_TO_ENUM:
            raise ValueError(f'Invalid dataset_name {dataset_name}')

        self.dataset_name = dataset_name
        self.attribution_files_train_path = attribution_files_train_path
        self.attribution_files_validation_path = attribution_files_validation_path
        self.attribution_files_test_path = attribution_files_test_path
        self.save_image = save_image
        self.num_saved_images = num_saved_images
        self.index = 0
        self.percentile = percentile
        self.roar = roar
        self.non_perturbed_testset = non_perturbed_testset

        self.dataset_class = dataset_factory.get_dataset_class(dataset_name=dataset_name)
        self.mean = self.dataset_class.mean
        self.std = self.dataset_class.std
        self.demean = [-m / s for m, s in zip(self.mean, self.std)]
        self.destd = [1 / s for s in self.std]

        self.train_normalize_transform = self.dataset_class.get_train_transform(enable_augmentation=False)
        self.evaluation_normalize_transform = self.dataset_class.get_validation_transform()
        # Used for visualization of preprocessed images.
        self.denormalize_transform = torchvision.transforms.Normalize(self.demean, self.destd)

        # Note - For training, we do not apply augmentation transform.
        # First, image is loaded, most/least important pixels are removed and then augmentations are applied.
        self.training_images_dataset = torchvision.datasets.ImageFolder(root=image_files_train_path,
                                                                        transform=torchvision.transforms.ToTensor())
        self.validation_images_dataset = torchvision.datasets.ImageFolder(root=image_files_validation_path,
                                                                          transform=self.evaluation_normalize_transform)
        self.test_images_dataset = torchvision.datasets.ImageFolder(root=image_files_test_path,
                                                                    transform=self.evaluation_normalize_transform)

        self.training_attribution_map_dataset = torchvision.datasets.ImageFolder(
            root=attribution_files_train_path,
            transform=torchvision.transforms.Compose(
                [torchvision.transforms.Grayscale(num_output_channels=3), torchvision.transforms.ToTensor()]))
        self.validation_attribution_map_dataset = torchvision.datasets.ImageFolder(
            root=attribution_files_validation_path,
            transform=torchvision.transforms.Compose(
                [torchvision.transforms.Grayscale(num_output_channels=3), torchvision.transforms.ToTensor()]))
        self.test_attribution_map_dataset = torchvision.datasets.ImageFolder(
            root=attribution_files_test_path,
            transform=torchvision.transforms.Compose(
                [torchvision.transforms.Grayscale(num_output_channels=3), torchvision.transforms.ToTensor()]))

        self.mode = 'training'

    def __getitem__(self, index):
        if self.mode == 'training':
            image, label = self.training_images_dataset[index]
            attribution_map, label = self.training_attribution_map_dataset[index]
            mean = self.mean
        elif self.mode == 'validation':
            image, label = self.validation_images_dataset[index]
            attribution_map, label = self.validation_attribution_map_dataset[index]
            mean = [0, 0, 0]
        else:
            image, label = self.test_images_dataset[index]
            attribution_map, label = self.test_attribution_map_dataset[index]
            mean = [0, 0, 0]  # validation and training images already are normalized.

        # Below code is left intentionally for one to quickly check if input data to model is correct.
        if self.save_image:
            T.ToPILImage()(image).save(f'tmp_imgs/input_{self.index}.jpg')  # only for training, for validation/test, denormalize first.
        image = np.array(image)
        attribution_map = np.max(attribution_map.numpy(), axis=0, keepdims=True)
        if self.non_perturbed_testset:
            if self.mode == 'training':
                image = roar_core.remove(image, attribution_map, mean, self.percentile, keep=not self.roar, gray=True)
        else:
            image = roar_core.remove(image, attribution_map, mean, self.percentile, keep=not self.roar, gray=True)

        if self.save_image:
            T.ToPILImage()(image.transpose(1, 2, 0).astype(np.uint8)).save(f'tmp_imgs/pert_input_{self.index}.jpg')

        if self.mode == 'training':
            # Do augmentation(randomscale/randomcrop) transform only after removal of pixels is done.
            image = image.transpose(1, 2, 0)  # PIL needs HXWX3, converting from 3xHxW .
            image = self.train_normalize_transform(Image.fromarray((image * 255).astype(np.uint8)))

        if self.save_image:
            T.ToPILImage()(self.denormalize_transform(image)).save(f'tmp_imgs/augmented_{self.index}.jpg')

        # increment index for saved images
        self.index = (self.index+1)%self.num_saved_images
        return image, label

    def __len__(self):
        if self.mode == 'training':
            return self.train_dataset_size
        elif self.mode == 'validation':
            return self.val_dataset_size
        else:
            return self.test_dataset_size

    def get_train_dataloader(self, data_args) -> DataLoader:
        self.mode = 'training'
        # Deepcopy ensures any changes to mode variable will not influence this dataloader
        return torch.utils.data.DataLoader(deepcopy(self),
                                           batch_size=data_args['batch_size'],
                                           shuffle=data_args['shuffle'],
                                           num_workers=get_cores_count())

    def get_validation_dataloader(self, data_args) -> DataLoader:
        self.mode = 'validation'
        return torch.utils.data.DataLoader(deepcopy(self),
                                           batch_size=data_args['batch_size'],
                                           shuffle=data_args['shuffle'],
                                           num_workers=get_cores_count())

    def get_test_dataloader(self, data_args):
        self.mode = 'test'
        return torch.utils.data.DataLoader(deepcopy(self),
                                           batch_size=data_args['batch_size'],
                                           shuffle=data_args['shuffle'],
                                           num_workers=get_cores_count())

    @property
    def classes(self):
        return self.training_attribution_map_dataset.classes

    @property
    def train_dataset_size(self):
        return len(self.training_images_dataset)

    @property
    def val_dataset_size(self):
        return len(self.validation_images_dataset)

    @property
    def test_dataset_size(self):
        return len(self.test_images_dataset)

    # ToDo Add debug method for Compound Image Folder Dataset.


class AttributionMapDataset(torch.utils.data.Dataset):
    """
    To load attribution maps as dataset.
    """

    def __init__(self,
                 attribution_files_train_path,
                 attribution_files_validation_path,
                 attribution_files_test_path,
                 percentile,
                 save_image=True,
                 num_saved_images=10,
                 ):
        """

        Args:
            attribution_files_train_path:
            attribution_files_validation_path:
            attribution_files_test_path:
        """

        self.attribution_files_train_path = attribution_files_train_path
        self.attribution_files_validation_path = attribution_files_validation_path
        self.attribution_files_test_path = attribution_files_test_path
        self.percentile = percentile
        self.save_image = save_image
        self.num_saved_images = num_saved_images
        self.index = 0

        self.training_attribution_map_dataset = torchvision.datasets.ImageFolder(
            root=attribution_files_train_path,
            transform=torchvision.transforms.Compose(
                [torchvision.transforms.Grayscale(num_output_channels=3), torchvision.transforms.ToTensor()]))
        self.validation_attribution_map_dataset = torchvision.datasets.ImageFolder(
            root=attribution_files_validation_path,
            transform=torchvision.transforms.Compose(
                [torchvision.transforms.Grayscale(num_output_channels=3), torchvision.transforms.ToTensor()]))
        self.test_attribution_map_dataset = torchvision.datasets.ImageFolder(
            root=attribution_files_test_path,
            transform=torchvision.transforms.Compose(
                [torchvision.transforms.Grayscale(num_output_channels=3), torchvision.transforms.ToTensor()]))

        self.mode = 'training'

    def __getitem__(self, index):
        if self.mode == 'training':
            attribution_map, label = self.training_attribution_map_dataset[index]
        elif self.mode == 'validation':
            attribution_map, label = self.validation_attribution_map_dataset[index]
        else:
            attribution_map, label = self.test_attribution_map_dataset[index]

        # perturb the attribution map
        # TODO add KAR version
        gray_img_tensor, _ = torch.max(attribution_map, axis=0, keepdims=True)
        gray_img_tensor = gray_img_tensor.squeeze()
        num_pixels = torch.numel(gray_img_tensor)
        num, _ = torch.topk(gray_img_tensor.flatten(), int(num_pixels * self.percentile / 100))
        attribution_map[:, gray_img_tensor < num[-1]] = 0

        # Below code is left intentionally for one to quickly check if input data to model is correct.
        if self.save_image:
            T.ToPILImage()(attribution_map).save(f'tmp_imgs/perturbed_attribution_map_{self.index}.jpg')

        # increment index for saved images
        self.index = (self.index+1)%self.num_saved_images
        return attribution_map, label

    def __len__(self):
        if self.mode == 'training':
            return self.train_dataset_size
        elif self.mode == 'validation':
            return self.val_dataset_size
        else:
            return self.test_dataset_size

    def get_train_dataloader(self, data_args) -> DataLoader:
        self.mode = 'training'
        # Deepcopy ensures any changes to mode variable will not influence this dataloader
        return torch.utils.data.DataLoader(deepcopy(self),
                                           batch_size=data_args['batch_size'],
                                           shuffle=data_args['shuffle'],
                                           num_workers=get_cores_count())

    def get_validation_dataloader(self, data_args) -> DataLoader:
        self.mode = 'validation'
        return torch.utils.data.DataLoader(deepcopy(self),
                                           batch_size=data_args['batch_size'],
                                           shuffle=data_args['shuffle'],
                                           num_workers=get_cores_count())

    def get_test_dataloader(self, data_args):
        self.mode = 'test'
        return torch.utils.data.DataLoader(deepcopy(self),
                                           batch_size=data_args['batch_size'],
                                           shuffle=data_args['shuffle'],
                                           num_workers=get_cores_count())

    @property
    def classes(self):
        return self.training_attribution_map_dataset.classes

    @property
    def train_dataset_size(self):
        return len(self.training_attribution_map_dataset)

    @property
    def val_dataset_size(self):
        return len(self.validation_attribution_map_dataset)

    @property
    def test_dataset_size(self):
        return len(self.test_attribution_map_dataset)


class ImageAndAttributionDataset(torch.utils.data.Dataset):
    def __init__(self, image_dataset, attribution_dataset):
        self.image_dataset = image_dataset
        self.attribution_dataset = attribution_dataset

    def __getitem__(self, index):
        image = self.image_dataset[index]
        attribution = self.attribution_dataset[index]
        return image, attribution

    def __len__(self):
        return len(self.image_dataset)

    def get_train_dataloader(self, data_args) -> DataLoader:
        self.image_dataset.mode = 'training'
        self.attribution_dataset.mode = 'training'
        # Deepcopy ensures any changes to mode variable will not influence this dataloader
        return torch.utils.data.DataLoader(deepcopy(self),
                                           batch_size=data_args['batch_size'],
                                           shuffle=data_args['shuffle'],
                                           num_workers=get_cores_count())

    def get_validation_dataloader(self, data_args) -> DataLoader:
        self.image_dataset.mode = 'validation'
        self.attribution_dataset.mode = 'validation'
        return torch.utils.data.DataLoader(deepcopy(self),
                                           batch_size=data_args['batch_size'],
                                           shuffle=data_args['shuffle'],
                                           num_workers=get_cores_count())

    def get_test_dataloader(self, data_args):
        self.image_dataset.mode = 'test'
        self.attribution_dataset.mode = 'test'
        return torch.utils.data.DataLoader(deepcopy(self),
                                           batch_size=data_args['batch_size'],
                                           shuffle=data_args['shuffle'],
                                           num_workers=get_cores_count())
