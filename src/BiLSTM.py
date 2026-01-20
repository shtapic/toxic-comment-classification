import numpy as np
import torch
import torch.nn as nn

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, hidden_dim, output_dim, n_layers, dropout, embed_dim, pretrained_embeddings=None):
        super().__init__()
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, padding_idx=0)
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.lstm = nn.LSTM(embed_dim, hidden_size=hidden_dim, num_layers=n_layers,
                            bidirectional=True, dropout=dropout, batch_first=True)
        self.clf = nn.Linear(hidden_dim * 2, output_dim)


    def forward(self, x):
        embedded = self.embedding(x)
        lstm, _ = self.lstm(embedded)
        not_padded, _ = torch.max(lstm, dim=1)
        out = self.clf(not_padded)
        return torch.sigmoid(out)
    