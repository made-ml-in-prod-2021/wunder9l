from typing import Tuple

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
    ):
        super(PretrainedRNN, self).__init__()
        tokens_count, dim = pretrained_vectors.shape
        self.pretrained_embeddings = nn.Embedding(
            tokens_count,
            dim,
            padding_idx=padding_idx,
        )
        self.hidden_size = hidden_size
        self.pretrained_embeddings.weight.data.copy_(pretrained_vectors)
        self.pretrained_embeddings._fill_padding_idx_with_zero()
        self.rnn = nn.RNN(
            input_size=dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )

    def forward(
        self, tokens: torch.Tensor, hidden_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns output, hidden_state"""
        assert tokens.shape[0] == hidden_state.shape[0], "Batch size must be equal"
        x = self.pretrained_embeddings[tokens]
        return self.rnn.forward(x, hidden_state)


def make_rnn_model(vectors: torch.Tensor, padding_idx: int, args: ModelRNNArgs) -> nn.Module:
    rnn = PretrainedRNN(
        pretrained_vectors=vectors,
        padding_idx=padding_idx,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
    )
    return rnn
