import torch
from torch import nn


class CNNColorDetector(nn.Module):

    def __init__(self, color_list, redundant_channels: int = 0, background_pixel=(0, 0, 0)):
        super(CNNColorDetector, self).__init__()

        # trick to append background pixel internally
        color_list = list(color_list)
        color_list.append(background_pixel)
        self.color_list = tuple(color_list)

        assert redundant_channels >= 0, 'num of redundant channels cannot be negative'
        self.redundant_channels = redundant_channels
        self.num_colors = len(self.color_list)
        self.first_stage = nn.Conv2d(in_channels=3, out_channels=9 * self.num_colors, kernel_size=1)
        self.second_stage = nn.Conv2d(in_channels=9 * self.num_colors, out_channels=6 * self.num_colors, kernel_size=1)
        self.third_stage = nn.Conv2d(in_channels=6 * self.num_colors, out_channels=3 * self.num_colors, kernel_size=1)
        self.sum_stage = nn.Conv2d(in_channels=3 * self.num_colors, out_channels=self.num_colors, kernel_size=1)

        # final stage removes the background color detector and add redundant channels
        self.sum_and_redundant_stage = nn.Conv2d(
            in_channels=self.num_colors, out_channels=self.num_colors + self.redundant_channels - 1, kernel_size=1)

    def forward(self, x):
        y = torch.relu(self.first_stage(x))
        y = torch.relu(self.second_stage(y))
        y = torch.relu(self.third_stage(y))
        y = torch.relu(self.sum_stage(y))
        y = torch.relu(self.sum_and_redundant_stage(y))
        return y


class PartialNonUniformCnnMultiColorAccumulator(nn.Module):

    def __init__(self, num_colors, redundant_channels: int = 0, random_expand_to: int = 3):
        super(PartialNonUniformCnnMultiColorAccumulator, self).__init__()
        assert redundant_channels >= 0, 'num of redundant channels need to be non negative'
        self.redundant_channels = redundant_channels
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=num_colors + redundant_channels,
                out_channels=num_colors,
                kernel_size=3,
                stride=3,
                padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_colors, out_channels=num_colors * random_expand_to, kernel_size=3, stride=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_colors * random_expand_to, out_channels=num_colors, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=num_colors, out_channels=num_colors * random_expand_to, kernel_size=3, stride=3, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_colors * random_expand_to, out_channels=num_colors, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_colors, out_channels=num_colors * random_expand_to, kernel_size=3, stride=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_colors * random_expand_to, out_channels=num_colors, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_colors, out_channels=num_colors * random_expand_to, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_colors * random_expand_to, out_channels=num_colors, kernel_size=1, stride=1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.layers(x)
        return x.view(x.size(0), -1)


class IdentityMLP(nn.Module):

    def __init__(self, input_size):
        super(IdentityMLP, self).__init__()
        self.fc = nn.Linear(input_size, input_size)

    def forward(self, x):
        out = self.fc(x)
        return out


class SyntheticModel(nn.Module):
    # use dummy color list for now
    def __init__(self, color_list=(0, 0, 0, 0), redundant_channels: int = 1, random_expand_to: int = 3):
        super(SyntheticModel, self).__init__()
        self.color_detector = CNNColorDetector(color_list=color_list,
                                               redundant_channels=redundant_channels)
        self.accumulator = PartialNonUniformCnnMultiColorAccumulator(num_colors=len(color_list),
                                                                     redundant_channels=redundant_channels,
                                                                     random_expand_to=random_expand_to)
        self.mlp = IdentityMLP(input_size=len(color_list))

        # from src.models.classification.PytorchCifarResnet import ResNet8
        # self.model = ResNet8()
        print(self)

    def forward(self, x):
        y = self.color_detector(x)
        y = self.accumulator(y)
        y = self.mlp(y)
        # y = self.model(x)
        return y
