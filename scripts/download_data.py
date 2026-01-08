from src.utils.download import download_text

URL = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"


if __name__ == "__main__":
    path = download_text(URL)
    print("Saved to:", path)


