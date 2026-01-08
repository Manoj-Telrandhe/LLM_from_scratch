import torch

from src.utils.download import download_text
from src.data_pipeline.dataset import create_dataloader
from src.model.token_pos_embed import create_token_pos_embeddings
from src.config import GPT_CONFIG_124M

URL = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
cfg = GPT_CONFIG_124M

if __name__ == "__main__":
    # Download & load text
    file_path = download_text(URL)
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    
    # Test this by creating a dataloader instance with batch size = 1
    # Create dataloader
    dataloader = create_dataloader(
        raw_text,
        batch_size=1,
        max_length=cfg["context_length"],
        stride=cfg["context_length"],
        shuffle=False,
    )

    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Inputs shape:\n", inputs.shape)   # 2D tensor
    print("Token IDs: \n", inputs)

    # Create embedding layers
    token_embedding_layer, pos_embedding_layer = create_token_pos_embeddings()

    # Token embeddings
    token_embeddings = token_embedding_layer(inputs)
    print("Token embeddings shape:", token_embeddings.shape)  # 3D tensor bcoz each token ID get mapped to 768 embed dim

    # Positional embeddings
    pos_embeddings = pos_embedding_layer(
        torch.arange(cfg["context_length"])
    )
    print("Positional embeddings shape:", pos_embeddings.shape)

    # Input embeddings (token + position)
    input_embeddings = token_embeddings + pos_embeddings
    print("Input embeddings shape:", input_embeddings.shape)
    print("Input Embeddings:\n", input_embeddings)
