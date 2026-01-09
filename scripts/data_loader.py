import torch
from src.data_pipeline.dataloader import create_dataloader
from src.config import GPT_CONFIG_124M
from src.utils.download import download_text
from src.tokenizer.bpe_tokenizer import my_tokenizer

def create_train_val_loaders(text_data, tokenizer, batch_size=2):
    
    
    token_ids = tokenizer.encode(text_data, allowed_special={"<|endoftext|>"})
    total_tokens = len(token_ids)

    train_ratio = 0.90
    train_tokens = int(train_ratio * total_tokens)
    val_tokens = total_tokens - train_tokens
    context_length = GPT_CONFIG_124M["context_length"]
    
    # Sanity check # if tokens are less than context length
    if train_tokens <= context_length:
        print("⚠️ Not enough tokens for the training loader. "
            "Try to lower the `GPT_CONFIG_124M['context_length']` or "
            "increase the `training_ratio`")

    if val_tokens <= context_length:
        print("⚠️ Not enough tokens for the validation loader. "
              "Try to lower the `GPT_CONFIG_124M['context_length']` or "
             "decrease the `training_ratio`")
     
    # Split the text
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_text = text_data[:split_idx]
    val_text = text_data[split_idx:]


    torch.manual_seed(123)

    train_loader = create_dataloader(
        txt=train_text,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length= GPT_CONFIG_124M["context_length"],
        stride= GPT_CONFIG_124M["context_length"],
        shuffle=True,
        drop_last=True,
    )

    val_loader = create_dataloader(
        txt=val_text,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length= GPT_CONFIG_124M["context_length"],
        stride= GPT_CONFIG_124M["context_length"],
        shuffle=False,
        drop_last=True,
    )

    return train_loader, val_loader


# ---------------- SANITY TEST ----------------
if __name__ == "__main__":


    URL = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
    text_data = download_text(URL)  

    train_loader, val_loader = create_train_val_loaders(text_data)

    print("Train loader:")
    for x, y in train_loader:
        print("Input shape :", x.shape)
        print("Target shape:", y.shape)
        break  # only one batch

    print("\nValidation loader:")
    for x, y in val_loader:
        print("Input shape :", x.shape)
        print("Target shape:", y.shape)
        break
