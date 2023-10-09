import torch.nn as nn

class CoordinateExtractor(nn.Module):
    def __init__(self):
        super(CoordinateExtractor, self).__init__()
        self.coordinate_feature_extractor = nn.Sequential()
        # group 1
        self.coordinate_feature_extractor.add_module('conv1', nn.Conv2d(
            1, 10, 3, stride=1, padding=1))  # [b, 10, 224, 224]
        self.coordinate_feature_extractor.add_module('bn1', nn.BatchNorm2d(10))
        self.coordinate_feature_extractor.add_module('relu1', nn.ReLU())
        # group 2
        self.coordinate_feature_extractor.add_module('conv2', nn.Conv2d(
            10, 32, 3, stride=2, padding=1))  # [b, 32, 112, 112]
        self.coordinate_feature_extractor.add_module('bn2', nn.BatchNorm2d(32))
        self.coordinate_feature_extractor.add_module('relu2', nn.ReLU())
        # group 3
        self.coordinate_feature_extractor.add_module('conv3', nn.Conv2d(
            32, 64, 3, stride=2, padding=1))  # [b, 64, 56, 56]
        self.coordinate_feature_extractor.add_module('bn3', nn.BatchNorm2d(64))
        self.coordinate_feature_extractor.add_module('relu3', nn.ReLU())
        # group 4
        self.coordinate_feature_extractor.add_module('conv4', nn.Conv2d(
            64, 128, 3, stride=2, padding=1))  # [b, 128, 28, 28]
        self.coordinate_feature_extractor.add_module(
            'bn4', nn.BatchNorm2d(128))
        self.coordinate_feature_extractor.add_module('relu4', nn.ReLU())
        # group 5
        self.coordinate_feature_extractor.add_module('conv5', nn.Conv2d(
            128, 256, 3, stride=2, padding=1))  # [b, 256, 14, 14]
        self.coordinate_feature_extractor.add_module(
            'bn5', nn.BatchNorm2d(256))
        self.coordinate_feature_extractor.add_module('relu5', nn.ReLU())
        # group 6
        self.coordinate_feature_extractor.add_module('conv6', nn.Conv2d(
            256, 256, 3, stride=2, padding=1))  # [b, 256, 7, 7]
        self.coordinate_feature_extractor.add_module(
            'bn6', nn.BatchNorm2d(256))
        self.coordinate_feature_extractor.add_module('relu6', nn.ReLU())

    def forward(self, x):
        return self.coordinate_feature_extractor(x)

class PersonFeatureExtractor(nn.Module):
    def __init__(self, in_channel=4352, out_channel=1024):
        super(PersonFeatureExtractor, self).__init__()
        self.person_feature_extractor = nn.Sequential()
        # group 1
        self.person_feature_extractor.add_module('conv1', nn.Conv2d(
            in_channel, 2048, 3, stride=2))  # [b, 2048, 3, 3]
        self.person_feature_extractor.add_module('bn1', nn.BatchNorm2d(2048))
        self.person_feature_extractor.add_module('relu1', nn.ReLU())
        # group 2
        self.person_feature_extractor.add_module('conv2', nn.Conv2d(
            2048, out_channel, 3))  # [b, 1024, 1, 1]
        self.person_feature_extractor.add_module('bn2', nn.BatchNorm2d(out_channel))
        self.person_feature_extractor.add_module('relu2', nn.ReLU())

    def forward(self, x):
        return self.person_feature_extractor(x)

