import torch

from src.tokenizer.bpe_tokenizer import my_tokenizer
from src.model.gpt_model import GPTModel
from src.generate.generate import generate_text_sample
from src.config import GPT_CONFIG_124M

# Device handling (IMPORTANT)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    tokenizer = my_tokenizer()

    start_context = "Hello, I am"
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    print("encoded_tensor.shape:", encoded_tensor.shape)
    print("encoded input tensor:", encoded_tensor)

    model = GPTModel(GPT_CONFIG_124M)
    model.eval()

    out = generate_text_sample(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=6,
        context_size=GPT_CONFIG_124M["context_length"]
    )

    print("Output:", out)
    print("Output length:", len(out[0]))

    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print("Decoded text:")
    print(decoded_text)
