import time
import torch

# ---- Config ----
from src.config import GPT_CONFIG_124M

# ---- Data ----
from src.utils.download import download_text  # if you already created this
from scripts.data_loader import create_train_val_loaders

# ---- Tokenizer ----
from src.tokenizer.bpe_tokenizer import my_tokenizer

# ---- Model ----
from src.model.gpt_model import GPTModel

# ---- Training ----
from src.training.train import train_model_simple


def main():
    # ---------------- Device ----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---------------- Reproducibility ----------------
    torch.manual_seed(123)

    # ---------------- Load data ----------------
    URL = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
    text_data = download_text(URL)   # returns full text string

    # ---------------- Tokenizer ----------------
    tokenizer = my_tokenizer()

    # ---------------- Dataloaders ----------------
    train_loader, val_loader = create_train_val_loaders(text_data, tokenizer=tokenizer, batch_size=2)

    # ---------------- Model ----------------
    model = GPTModel(GPT_CONFIG_124M).to(device)

    # ---------------- Optimizer ----------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.1)

    # ---------------- Training ----------------
    start_time = time.time()

    num_epochs = 1
    train_losses, val_losses, tokens_seen = train_model_simple(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        eval_freq=5,
        eval_iter=5, 
        start_context="Every effort moves you",
        tokenizer=tokenizer,
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"\nTraining completed in {execution_time_minutes:.2f} minutes.")


if __name__ == "__main__":
    main()
