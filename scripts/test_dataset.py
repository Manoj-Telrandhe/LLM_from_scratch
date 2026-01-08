from src.utils.download import download_text
from src.data_pipeline.dataset import create_dataloader
from src.config import GPT_CONFIG_124M

# URL for text data
URL = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

if __name__ == "__main__":
    # Download text
    file_path = download_text(URL)

    # Load text
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Create dataloader (sanity test config) 
    # test the dataloader with batch size = 2
    dataloader = create_dataloader(
        raw_text,
        batch_size=2,
        max_length=4,
        stride=1,
        shuffle=False,
    )

    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Inputs:\n", inputs)
    print("\nTargets:\n", targets)



 