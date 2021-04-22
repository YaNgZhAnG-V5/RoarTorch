from pprint import pprint

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

from src.utils.sysutils import get_cores_count


# ToDo Move denormalization transform to transforms package
class Denormalize(object):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.demean = [-m / s for m, s in zip(mean, std)]
        self.std = std
        self.destd = [1 / s for s in std]
        self.inplace = inplace

    def __call__(self, tensor):
        tensor.normalize(tensor, self.demean, self.destd, self.inplace)
        # clamp to get rid of numerical errors
        return torch.clamp(tensor, 0.0, 1.0)

# ToDo - Design BaseDataset class that others should implement for integrating their own dataset
class CARS:
    # RGB Order
    # imagenet mean
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    def __init__(self,
                 dataset_args,
                 train_data_args,
                 val_data_args,
                 ):
        """
        use_random_flip not used.
        """

        self.cpu_count = get_cores_count()
        self.train_data_args = train_data_args
        self.val_data_args = val_data_args

        dataset_dir = dataset_args['dataset_dir']
        split_ratio = dataset_args.get('split_ratio', 7.0 / 8.0)
        assert split_ratio < 1.0, 'train set should be split into train and cross-validation set.'

        # Use augmentations for training models but not during generating dataset.
        self.train_transform = self.get_train_transform(
            enable_augmentation=train_data_args.get('enable_augmentation', False)
        )
        self.validation_transform = self.get_validation_transform()
        # Normalization transform does (x - mean) / std
        # To denormalize use mean* = (-mean/std) and std* = (1/std)
        self.demean = [-m / s for m, s in zip(self.mean, self.std)]
        self.destd = [1 / s for s in self.std]
        self.denormalization_transform = torchvision.transforms.Normalize(self.demean, self.destd, inplace=False)

        self.trainset = torchvision.datasets.ImageFolder(root=os.path.join(dataset_dir, 'train'), transform=self.train_transform)
        self.validationset = torchvision.datasets.ImageFolder(root=os.path.join(dataset_dir, 'train'), transform=self.validation_transform)

        # Split train data into training and cross validation dataset using 9:1 split ration
        training_indices, validation_indices = self._uniform_train_val_split(self.trainset.targets, split_ratio)
        self.trainset = torch.utils.data.Subset(self.trainset, training_indices)
        self.validationset = torch.utils.data.Subset(self.validationset, validation_indices)

        self.testset = torchvision.datasets.ImageFolder(root=os.path.join(dataset_dir,'test'), transform=self.validation_transform)

    @classmethod
    def get_train_transform(cls, enable_augmentation=False):
        """"""
        if enable_augmentation:
            normalize_transform = torchvision.transforms.Compose(
                [torchvision.transforms.RandomCrop(224, padding=4),
                 torchvision.transforms.RandomHorizontalFlip(),
                 torchvision.transforms.ToTensor(),
                 torchvision.transforms.Normalize(cls.mean, cls.std)]
            )
            print('Added random crop with 4 padding and random horizontal flips augmentation done')
        else:
            normalize_transform = torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor(),
                 torchvision.transforms.Normalize(cls.mean, cls.std)]
            )
            print('No augmentation done')
        return normalize_transform

    @classmethod
    def get_validation_transform(cls):
        return torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize(cls.mean, cls.std)]
        )

    @property
    def train_dataloader(self) -> DataLoader:
        return torch.utils.data.DataLoader(self.trainset,
                                           batch_size=self.train_data_args['batch_size'],
                                           shuffle=self.train_data_args['shuffle'],
                                           pin_memory=True,
                                           num_workers=get_cores_count())

    @property
    def validation_dataloader(self) -> DataLoader:
        return torch.utils.data.DataLoader(self.validationset,
                                           batch_size=self.train_data_args['batch_size'],
                                           shuffle=self.train_data_args['shuffle'],
                                           pin_memory=True,
                                           num_workers=get_cores_count())

    @property
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.testset,
                                           batch_size=self.val_data_args['batch_size'],
                                           shuffle=self.val_data_args['shuffle'],
                                           pin_memory=True,
                                           num_workers=get_cores_count())

    def imshow(self, img):
        # clamp to get rid of numerical errors
        img = torch.clamp(self.denormalize(img), 0.0, 1.0)  # denormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    def debug(self):
        # get some random training images
        data_iter = iter(self.train_dataloader)
        images, labels = data_iter.next()

        # show images
        self.imshow(torchvision.utils.make_grid(images))

        # print labels
        pprint(' '.join('%s' % self.classes[labels[j]] for j in range(len(images))))

    def denormalize(self, x):
        return self.denormalization_transform(x)

    @property
    def train_dataset_size(self):
        return len(self.trainset)

    @property
    def val_dataset_size(self):
        return len(self.validationset)

    @property
    def test_dataset_size(self):
        return len(self.testset)

    @property
    def classes(self):
        return [i for i in range(196)]

    def _uniform_train_val_split(self, targets, split_ratio):
        if type(targets) == list:
            targets = np.array(targets)
            labels = targets
        elif type(targets) == torch.tensor or type(targets) == torch.Tensor:
            labels = targets.numpy()
        training_indices = []
        validation_indices = []
        for i in range(len(self.classes)):
            label_indices = np.argwhere(labels == i)
            samples_per_label = int(split_ratio * len(label_indices))
            training_label_indices = label_indices[:samples_per_label]
            validation_label_indices = label_indices[samples_per_label:]
            training_indices.extend(training_label_indices.squeeze().tolist())
            validation_indices.extend(validation_label_indices.squeeze().tolist())
            assert not set(training_label_indices.ravel().tolist()) & set(validation_label_indices.ravel().tolist())

        assert not set(training_indices) & set(validation_indices)
        return training_indices, validation_indices

    def pos_neg_balance_weights(self):
        return torch.tensor([1.0] * len(self.classes))


def get_cars_object():
    dataset_args = dict(
      dataset_dir='./cars',
      split_ratio=0.9,
    )

    train_data_args = dict(
        batch_size=8,
        shuffle=False,
    )

    val_data_args = dict(
        batch_size=train_data_args['batch_size'] * 4,
        shuffle=False,
        validate_step_size=1,
    )
    dataset = CARS(dataset_args, train_data_args, val_data_args)
    return dataset


if __name__ == '__main__':
    dataset = get_cars_object()
    dataset.debug()

    print('Length of training set = ', len(dataset.trainset))
    print('Length of validation set = ', len(dataset.validationset))
    print('Length of test set = ', len(dataset.testset))