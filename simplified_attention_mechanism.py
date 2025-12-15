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


# Visualization
# Create 3D plot with vectors from origin to each point, using different colors
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define a list of colors for the vectors
colors = ['r', 'g', 'b', 'c', 'm', 'y']

# Plot each vector with a different color and annotate with the corresponding word
for (x, y, z, word, color) in zip(x_coords, y_coords, z_coords, words, colors):
    # Draw vector from origin to the point (x, y, z) with specified color and smaller arrow length ratio
    ax.quiver(0, 0, 0, x, y, z, color=color, arrow_length_ratio=0.05)
    ax.text(x, y, z, word, fontsize=10, color=color)

# Set labels for axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set plot limits to keep arrows within the plot boundaries
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_zlim([0, 1])

plt.title('3D Plot of Word Embeddings with Colored Vectors')
plt.show()


# Attention Scores
attn_scores = torch.empty(6, 6)

for i, x_i in enumerate(inputs):
  for j, x_j in enumerate(inputs):
    attn_scores[i, j] = torch.dot(x_i, x_j)

print(attn_scores)

# When computing the preceding attention score tensor, we used for-loops in Python.
# However, for-loops are generally slow, and we can achieve the same results using matrix multiplication:
attn_scores = inputs @ inputs.T
print(attn_scores)
# The matrix multiplication much more efficient than two for loops


# Normalization 
# Normal Theorotical Softmax formula may encounter numerical instablity problems, such as overflow and underflow when dealing with large or small input values respectively.
# Therefore, in practice it's advisable to use PyTorch implementation of softmax, which has been extensively optimized for performance:
attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)
# By setting the dim=-1, we are instructing the softmax function to apply the normalization along the last dimension of the attn_scores tensor.


# Final step we will calculate the context vector by multiplying the inputs embeddings with their respective attention weights
# context vector is enriched embedding vector
all_context_vecs = attn_weights @ inputs
print(all_context_vecs)















