import torch
input_ids = torch.tensor([2, 3, 5, 1])

vocab_size = 6
output_dim = 3

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

print(embedding_layer)
print(embedding_layer.weight)

# convert single token ID into a three-dim embd vector
print(embedding_layer(torch.tensor(3)))

# now apply it to a input_ids (torch.tensor([2, 3, 5, 1])).
print(embedding_layer(input_ids))

# now consider more realistic and useful ebedding sizes and encode the input tokens into a 256-dimensional vector representation.
# take the vocab size gpt-2 vocab size
vocab_size = 50257
output_dim = 256

#It takes two input.....torch.nn.Embedding(num_embeddings, embedding_dim)
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

print(token_embedding_layer.weight)
