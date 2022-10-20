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
        """
        seq2seq transformer model
        :param d_spectrogram: dim of input spectrogram frames (no. of mel bins)
        :param d_model: feature dim of the encoder/decoder inputs
        :param n_token: number of output token types
        :param n_head: number of heads in multi-head attention
        :param n_hidden: number of hidden neurons in the feedforward network model
        :param n_encoder_layer: number of sub-encoder-layers in the encoder
        :param n_decoder_layer: number of sub-decoder-layers in the decoder
        :param dropout: the dropout value
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
        :param src: source sequence (batch size, seq. length, d_spectrogram)
        :param tgt: target sequence (batch size, seq. length)
        :return output: softmax distribution over a discrete vocabulary of events (batch size, seq. length, n_token)
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
        """
        initialize weights with uniform distribution in range [-0.1, 0.1]
        """
        initrange = 0.1
        self.src_embedder.weight.data.uniform_(-initrange, initrange)
        self.tgt_embedder.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def get_subsequent_mask(self, size: int) -> torch.tensor:
        """
        generates a target/subsequent mask
        :param size: target sequence length
        :return: mask of shape (size, size)
        """
        return torch.triu(torch.full((size, size), float("-inf")), diagonal=1)

    def get_padding_mask(self, x: torch.tensor, pad_token: int) -> torch.tensor:
        """
        generates a padding mask, tells the model to ignore padding tokens at the end of sequence
        :param x: input
        :param pad_token: id of padding token
        :return: mask with the value 'True' on positions with the padding token, 'False' otherwise
        """
        return x == pad_token


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        """
        injects information about the relative or absolute position of the tokens in the sequence
        source: https://github.com/pytorch/examples/blob/main/word_language_model/model.py
        :param d_model: the embed dim
        :param dropout: the dropout value
        :param max_len: the max. length of the incoming sequence
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
        :param x: the input sequence of shape (batch size, seq. length, embed dim)
        :return: position embedded sequence of shape (batch size, seq. length, embed dim)
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)
