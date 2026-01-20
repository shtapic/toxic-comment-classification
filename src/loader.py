import re
import numpy as np
from .preprocess_data import clean_text
import torch
from torch.utils.data import Dataset

def encode_text(text, max_length, word_to_idx):
    tokens = clean_text(text).split()
    indices = []
    
    unk_idx = word_to_idx.get('<UNK>', 1)
    pad_idx = word_to_idx.get('<PAD>', 0)

    for token in tokens:
        indices.append(word_to_idx.get(token, unk_idx))
    
    if len(indices) < max_length:
        indices += [pad_idx] * (max_length - len(indices))
    else:
        indices = indices[:max_length]
        
    return torch.LongTensor(indices)

def create_embedding_matrix_and_vocab(FastText_pipeline):
    fast_text_model = FastText_pipeline.named_steps['fasttext'].model
    word_to_idx = {word : i+2 for i, word in enumerate(fast_text_model.wv.index_to_key)}
    word_to_idx['<PAD>'] = 0
    word_to_idx['<UNK>'] = 1

    vocab_size = len(word_to_idx)
    embed_dim = fast_text_model.vector_size
    embedding_matrix = np.zeros((vocab_size, embed_dim))

    print(f"Vocab size: {vocab_size}, Embedding dim: {embed_dim}")

    for word, idx in word_to_idx.items():
        if idx == 0: continue # PAD -> zeros
        if idx == 1:
            embedding_matrix[idx] = fast_text_model.wv['jsdfhskdjfh'] 
        else:
            if word in fast_text_model.wv:
                embedding_matrix[idx] = fast_text_model.wv[word]

    return torch.FloatTensor(embedding_matrix), word_to_idx


class ToxictDataset(Dataset):
    def __init__(self, df, text_column, target_columns, max_length, word_to_idx):
        super().__init__()
        self.text = df[text_column].values
        self.targets = df[target_columns].values
        self.max_length = max_length
        self.word_to_idx = word_to_idx

    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, index):
        text = self.text[index]
        labels = self.targets[index]
        
        enc_text = encode_text(text, self.max_length, word_to_idx=self.word_to_idx)
        return enc_text, torch.FloatTensor(labels)