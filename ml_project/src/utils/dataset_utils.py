import collections


class FeaturesWithLabels(object):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    # def split(self, train_size):
