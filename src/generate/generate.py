import torch 
import torch.nn as nn     

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_text_sample(model, idx, max_new_tokens, context_size):
  model.device()
  idx = idx.to(device)
  # idx is (batch, n_tokens) array of indices in the current context
  for _ in range(max_new_tokens):

    # Crop current context if it exceeds the supported context size
    # E.g., if LLM supports only 5 tokens, and the context size is 10
    # then only the last 5 tokens are used as context
    idx_cond = idx[:, -context_size:]

    # Get the predictions
    with torch.no_grad():
      logits = model(idx_cond)  ## logits--> batch, n_tokens, vocab_size

    # Focus only on the last time step
    # (batch, n_tokens, vocab_Size) becomes (batch, vocab_size)
    logits = logits[:, -1, :]

    # Apply softmax to get the probabilities
    probs = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

    # Get the idx of the vocab entry with the highest probability value
    idx_next = torch.argmax(probs, dim=-1, keepdim=True)  # (batch, 1)

    # Append sampled index to the running sequence
    idx = torch.cat((idx, idx_next), dim=1)  #(batch, n_tokens+1)

  return idx
