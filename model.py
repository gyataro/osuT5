import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(
        self,
        d_spectrogram: int,
        d_model: int,
        n_token: int,
        n_encoder_layer: int,
        n_decoder_layer: int,
        n_head: int,
        n_hidden: int,
        dropout: float,
    ):
        """Seq2seq transformer model.

        Attributes:
            d_spectrogram: dim of input spectrogram frames (no. of mel bins).
            d_model: feature dim of the encoder/decoder inputs.
            n_token: number of output token types.
            n_head: number of heads in multi-head attention.
            n_hidden: number of hidden neurons in the feedforward network model.
            n_encoder_layer: number of sub-encoder-layers in the encoder.
            n_decoder_layer: number of sub-decoder-layers in the decoder.
            dropout: the dropout value.
        """
        super().__init__()
        self.model_type = "Transformer"
        self.src_embedder = nn.Linear(d_spectrogram, d_model, bias=False)
        self.tgt_embedder = nn.Embedding(n_token, d_model)
        self.pos_encoder = PositionalEncoder(d_model, dropout)
        self.transformer = nn.Transformer(
            d_model,
            n_head,
            n_encoder_layer,
            n_decoder_layer,
            n_hidden,
            dropout,
            batch_first=True,
        )
        self.linear = nn.Linear(d_model, n_token)

    def forward(
        self,
        src: torch.tensor,
        tgt: torch.tensor,
        tgt_mask: torch.tensor = None,
        tgt_pad_mask: torch.tensor = None,
    ) -> torch.tensor:
        """
        Args:
            src: Source sequence (batch size, seq. length, d_spectrogram).
            tgt: Target sequence (batch size, seq. length).

        Returns:
            Softmax distribution over a discrete vocabulary of events (batch size, seq. length, n_token).
        """
        src = self.src_embedder(src)
        src = self.pos_encoder(src)
        tgt = self.tgt_embedder(tgt)
        tgt = self.pos_encoder(tgt)

        output = self.transformer(
            src, tgt, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_pad_mask
        )
        output = self.linear(output)
        return F.log_softmax(output, dim=-1)

    def init_weights(self):
        """Initialize weights with uniform distribution in range [-0.1, 0.1]."""
        initrange = 0.1
        self.src_embedder.weight.data.uniform_(-initrange, initrange)
        self.tgt_embedder.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def get_subsequent_mask(self, size: int) -> torch.tensor:
        """Generate a target/subsequent mask of shape (size, size)."""
        return torch.triu(torch.full((size, size), float("-inf")), diagonal=1)

    def get_padding_mask(self, x: torch.tensor, pad_token: int) -> torch.tensor:
        """Generate a padding mask by marking positions with `pad_token` as `True`, `False` otherwise."""
        return x == pad_token


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        """Inject information about the relative or absolute position of the tokens in the sequence.

        Source: https://github.com/pytorch/examples/blob/main/word_language_model/model.py

        Attributes:
            d_model: the embed dim
            dropout: the dropout value
            max_len: the max. length of the incoming sequence
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: The input sequence of shape (batch size, seq. length, embed dim).

        Returns:
            A position embedded sequence of shape (batch size, seq. length, embed dim).
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)
