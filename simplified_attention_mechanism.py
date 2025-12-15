# IMPLEMENTING A SIMPLIFIED ATTENTION MECHANISM
# consider a following inout sentence, which has alaready been embedded into 3 dimensional vectors.

# We choose a small embedding dim for illustration purpose to ensure it fits on page withtout line brake.

import torch
# sentence = "Your journer starts with one step"
# Each row represents a word, and each column represents an embedding dimension
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)




