import argparse
import itertools
import os
import random
import time
from pprint import pformat

import texttable
import torch
import yaml
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

from src.utils import utils
from src.utils.sysutils import is_debug_mode

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from src.loss import utils as loss_utils
from src.models import utils as models_utils
from src.optimizer import utils as optimizer_utils
from src.roar import compound_image_folder_dataset, roar_core


def main():
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument("--config", type=str, default="config/roar_cifar10_resnet8.yml", help="Configuration file to use.")
    args = parser.parse_args()
    with open(args.config) as fp:
        cfg = yaml.load(fp, Loader=Loader)

    roar_core.validate_configuration(cfg, validate_attribution_methods=True)

    inputdir = os.path.join(cfg['outdir'], 'extract_cams')
    dataset_name = cfg['data']['dataset']
    eval_metric = cfg['retrain_cls']['metrics']
    attribution_methods = cfg['retrain_cls']['attribution_methods']
    percentiles = cfg['retrain_cls']['percentiles']
    non_perturbed_testset = cfg['retrain_cls']['non_perturbed_testset']
    save_image = cfg['retrain_cls']['save_debug_image']

    tt = texttable.Texttable()
    tt.header(['dataset', 'eval_metric', 'attribution_method', 'percentile', 'test_accuracy'])
    tt.set_cols_width([10, 10, 30, 10, 10])

    for percentile, attribution_method in itertools.product(percentiles, attribution_methods):
        # Output Directory
        models_savedir = os.path.join(cfg['outdir'], eval_metric)  # For saving results
        os.makedirs(models_savedir, exist_ok=True)

        # Create Compound Dataset from Image Dataset and attribution image dataset
        dataset_modes = ['train', 'validation', 'test']
        image_folder_paths = [os.path.join(inputdir, dataset_mode, 'input')
                              for dataset_mode in dataset_modes]
        attribution_paths = [os.path.join(inputdir, dataset_mode, attribution_method['name'])
                             for dataset_mode in dataset_modes]
        compound_dataset = compound_image_folder_dataset.CompoundImageFolderDataset(dataset_name,
                                                                                    image_folder_paths[0],
                                                                                    image_folder_paths[1],
                                                                                    image_folder_paths[2],
                                                                                    attribution_paths[0],
                                                                                    attribution_paths[1],
                                                                                    attribution_paths[2],
                                                                                    roar=eval_metric == 'roar',
                                                                                    percentile=percentile,
                                                                                    non_perturbed_testset=
                                                                                    non_perturbed_testset,
                                                                                    save_image=save_image,)

        # Create attribution map dataset using class label of attributed images
        if cfg['retrain_cls']['attribution_penalty'] is True:
            attribution_dataset = compound_image_folder_dataset.AttributionMapDataset(attribution_paths[0],
                                                                                      attribution_paths[1],
                                                                                      attribution_paths[2],
                                                                                      percentile=percentile)
        else:
            attribution_dataset = None

        # Retrain Model on Compound Dataset and Evaluate
        train_data_args = dict(
            batch_size=cfg['train_cls']['batch_size'],
            shuffle=True,
            enable_augmentation=True
        )

        val_data_args = dict(
            batch_size=train_data_args['batch_size'] * 4,
            shuffle=False,
            validate_step_size=1,
        )

        # Always retrain from start
        assert cfg['retrain_cls']['model'][
                   'pretrained_weights'] is None, 'Retraining should be done without pretrained weights'

        arguments = dict(
            cuda_device=cfg['cuda_device'],
            dataset=compound_dataset,
            non_perturbed_testset=non_perturbed_testset,
            attribution_beta=cfg['retrain_cls']['attribution_beta'],
            model_name_args=(attribution_method, percentile),
            train_data_args=train_data_args,
            val_data_args=val_data_args,
            model_args=cfg['retrain_cls']['model'],
            loss_args=cfg['retrain_cls']['loss'],
            optimizer_args=cfg['retrain_cls']['optimizer'],
            scheduler_args=cfg['retrain_cls']['scheduler'],
            outdir=models_savedir,
            nb_epochs=cfg['retrain_cls']['nb_epochs'],
            random_seed=random.randint(0, 1000),
            model_path=cfg['eval_trained_model']['model_path'],
        )

        final_roar_test_accuracy = train_and_evaluate_model(arguments)
        tt.add_row([dataset_name, eval_metric, attribution_method, percentile, final_roar_test_accuracy])
        print(tt.draw())


def train_and_evaluate_model(arguments):
    """
    Main Pipeline for training and cross-validation.
    """

    """ Setup result directory and enable logging to file in it """
    print('Arguments:\n{}'.format(pformat(arguments)))

    """ Set random seed throughout python"""
    utils.set_random_seed(random_seed=arguments['random_seed'])

    """ Set device - cpu or gpu """
    device = torch.device(arguments['cuda_device'] if torch.cuda.is_available() else "cpu")
    print(f'Using device - {device}')

    """ Load Compound Dataset """
    dataset = arguments['dataset']

    """ Load Model with weights(if available) """
    model: torch.nn.Module = models_utils.get_model(
        arguments.get('model_args'), device, dict(labels_count=len(dataset.classes))
    ).to(device)

    """ Create loss function """
    criterion = loss_utils.create_loss(arguments['loss_args'])

    """ Evaluate model on test set """
    model.load_state_dict(torch.load(arguments["model_path"]), strict=False)
    test_dataloader = dataset.get_test_dataloader(arguments['val_data_args'])  # batch size for val/test is same
    test_loss, test_accuracy = evaluate_single_class(device, model, test_dataloader, criterion)
    print(f'Accuracy of the network on the {dataset.test_dataset_size} test images: {test_accuracy} %%')
    return test_accuracy


def evaluate_single_class(device, model, dataloader, criterion):
    correct, total_samples = 0, 0
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader)):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total_samples
    total_loss /= total_samples
    return total_loss, accuracy


if __name__ == '__main__':
    main()
