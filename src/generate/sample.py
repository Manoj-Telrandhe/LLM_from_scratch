import torch     
import torch.nn as nn     
from src.generate.generate import generate_text_sample
from src.generate.helper import (text_to_token_ids, token_ids_to_text)

def generate_and_print_sample(model, tokenizer, device, start_context):
  model.eval()
  context_size = model.pos_emb.weight.shape[0]
  encoded = text_to_token_ids(start_context, tokenizer).to(device)
  with torch.no_grad():
    token_ids = generate_text_sample(model=model, idx=encoded,
                                     max_new_tokens=25, context_size=context_size)

  decoded_text = token_ids_to_text(token_ids, tokenizer)
  print(decoded_text.replace("\n", " "))  # Compact print format
  model.train()