from enum import Enum


class TokenizerType(Enum):
    WORD_PUNCTUATION = 1


class EModelType(Enum):
    RNN = 1


class ELossType(Enum):
    ENLLLoss = 1
    EBCEWithLogitsLoss = 2


class EOneBatchType(Enum):
    Common = 1
    Rnn = 2


class EOptimizerType(Enum):
    Adam = 1
    SGD = 2


class ELRSchedulerType(Enum):
    No = 0
    Step = 1


class EProgramMode(Enum):
    PrepareData = 1
    Train = 2
    Predict = 3
    Visualize = 4
