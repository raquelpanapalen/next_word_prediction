import math

import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        num_layers,
        dropout=0.2,
    ):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

        # self.init_weights()

    def forward(self, x):
        embedding = self.dropout(self.embedding(x))
        output, _ = self.lstm(embedding)
        output = self.fc(
            self.dropout(output[:, -1, :])
        )  # we take the last hidden state
        return output
