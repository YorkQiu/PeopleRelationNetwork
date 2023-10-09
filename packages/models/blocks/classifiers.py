import torch.nn as nn

class FeatureClassifier(nn.Module):
    def __init__(self, in_dim, out_dim=2, hidden_dim=512):
        super(FeatureClassifier, self).__init__()
        self.classifier = nn.Sequential()
        if hidden_dim == 0:
            self.classifier.add_module('fc1', nn.Linear(in_dim, out_dim))
        else:
            self.classifier.add_module('fc1', nn.Linear(in_dim, hidden_dim))
            self.classifier.add_module('relu1', nn.ReLU())
            self.classifier.add_module('dropoout1', nn.Dropout(0.5))
            self.classifier.add_module('fc2', nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        logit = self.classifier(x)
        return logit