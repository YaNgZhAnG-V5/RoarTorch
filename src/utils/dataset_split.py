import os
import shutil
import argparse


def split_data(root_path: str, val_ratio: float = 0.1, test_ratio: float = 0.1):
    """
    Given a root path, split the data into train, validation and test sets.
    Each subfolder in the root path is considered a class.
    """
    train_path = os.path.join(root_path, "train")
    val_path = os.path.join(root_path, "validation")
    test_path = os.path.join(root_path, "test")
    if not os.path.exists(train_path):
        os.mkdir(train_path)
    if not os.path.exists(val_path):
        os.mkdir(val_path)
    if not os.path.exists(test_path):
        os.mkdir(test_path)
    for method_name in os.listdir(root_path):
        if method_name in ["train", "validation", "test"]:
            continue
        os.mkdir(os.path.join(train_path, method_name))
        os.mkdir(os.path.join(val_path, method_name))
        os.mkdir(os.path.join(test_path, method_name))
        for class_name in os.listdir(os.path.join(root_path, method_name)):
            class_path = os.path.join(root_path, method_name, class_name)
            train_class_path = os.path.join(train_path, method_name, class_name)
            val_class_path = os.path.join(val_path, method_name, class_name)
            test_class_path = os.path.join(test_path, method_name, class_name)
            if not os.path.exists(train_class_path):
                os.mkdir(train_class_path)
            if not os.path.exists(val_class_path):
                os.mkdir(val_class_path)
            if not os.path.exists(test_class_path):
                os.mkdir(test_class_path)
            for i, file_name in enumerate(os.listdir(class_path)):
                if i < int(test_ratio * len(os.listdir(class_path))):
                    shutil.copy(
                        os.path.join(class_path, file_name), test_class_path
                    )
                elif i < int((test_ratio + val_ratio) * len(os.listdir(class_path))):
                    shutil.copy(
                        os.path.join(class_path, file_name), val_class_path
                    )
                else:
                    shutil.copy(
                        os.path.join(class_path, file_name), train_class_path
                    )
    print("Done splitting data")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='/home/yang/RoarTorch/runs/cars_vgg16/extract_cams/multicolor_sum/original_data/')
    parser.add_argument('--val-ratio', default=0.1, type=float)
    parser.add_argument('--test-ratio', default=0.1, type=float)
    args = parser.parse_args()
    split_data(args.data_root, args.val_ratio, args.test_ratio)

