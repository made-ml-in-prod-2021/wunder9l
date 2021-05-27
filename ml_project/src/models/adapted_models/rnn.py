from typing import Tuple, Optional

import torch
from torch import nn

from src.config.train.model import ModelRNNArgs


class PretrainedRNN(nn.Module):
    def __init__(
        self,
        pretrained_vectors: torch.Tensor,
        padding_idx: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_classes: int,
    ):
        """Adapted version of RNN.

        Consists of 3 parts: (pretrained) embeddings, RNN (form torch realization) and
        head with fully connected layers to predict classes.

        Args:
              **pretrained_vectors** - for embeddings

              **padding_idx** - index in pretrained_vectors that corresponds to padding token

              **hidden_size** - size of each hidden state

              **num_layers** - layers of RNN (see original docs)

              **dropout** - dropout in RNN (see original docs)

              **bidirectional** - bidirectional arg in RNN (see original docs)

              **num_classes** - number of classes in head (dim of output tensor)
        """
        super(PretrainedRNN, self).__init__()
        tokens_count, dim = pretrained_vectors.shape
        self.pretrained_embeddings = nn.Embedding(
            tokens_count,
            dim,
            padding_idx=padding_idx,
        )
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.pretrained_embeddings.weight.data.copy_(pretrained_vectors)
        self.pretrained_embeddings._fill_padding_idx_with_zero()
        self.rnn = nn.RNN(
            input_size=dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def forward(
        self, tokens: torch.Tensor, hidden_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of model: takes tokens, returns output, hidden_state

        Args:
            tokens: Tensor(seq_len x batch)

            hidden_state: Tensor(num_layers * num_directions, batch, hidden_size).
            Defaults to zero if not provided.

        Outputs:
            res - tensor (seq_len x batch x num_classes) - result for each token of each batch

            hidden - hidden_state of rnn for last token (seq_len[-1])
        """
        if hidden_state is not None:
            assert tokens.shape[1] == hidden_state.shape[1], "Batch size must be equal"
        x = self.pretrained_embeddings(tokens)
        rnn_output, hidden = self.rnn.forward(x, hidden_state)
        x = self.dropout_layer(rnn_output)
        x = self.fc(x)
        return x, hidden


def make_rnn_model(
    vectors: torch.Tensor, padding_idx: int, args: ModelRNNArgs
) -> nn.Module:
    rnn = PretrainedRNN(
        pretrained_vectors=vectors,
        padding_idx=padding_idx,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        num_classes=args.num_classes,
    )
    return rnn
