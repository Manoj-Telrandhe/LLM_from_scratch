import torch
from src.config import GPT_CONFIG_124M

cfg = GPT_CONFIG_124M


def create_token_pos_embeddings():
    """
    Creates token and positional embedding layers.
    """
    token_embedding_layer = torch.nn.Embedding(
        cfg["vocab_size"], cfg["emb_dim"]
    )

    pos_embedding_layer = torch.nn.Embedding(
        cfg["context_length"], cfg["emb_dim"]
    )

    return token_embedding_layer, pos_embedding_layer
