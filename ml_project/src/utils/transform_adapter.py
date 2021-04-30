from sklearn.base import BaseEstimator, TransformerMixin


class TransformCallbackAdapter(BaseEstimator, TransformerMixin):
    def __init__(self, callback):
        self.callback = callback

    # Return self nothing else to do here
    def fit(self, X, y=None):
        return self

    # Transformer method we wrote for this transformer
    def transform(self, X, y=None):
        return [self.callback(x) for x in X]
