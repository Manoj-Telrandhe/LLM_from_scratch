# Building a GPT from Scratch
This repository is built to deeply understand **how Large Language Models work from scratch**.


Instead of focusing on scale or performance, it breaks down core components such as tokenization, self-attention, multi-head attention, transformer blocks, and autoregressive text generation.


The goal is clarity: seeing how each part fits together to form a GPT-style language model.
___

## 📌 Project Workflow

This project is organized to help you understand each building block of a GPT-style language model independently, while still allowing everything to come together in a complete training pipeline.


The workflow follows the same logical order as a real LLM:

---

## Data Preparation & Tokenization

**Location:** `src/data_pipeline/` and `src/tokenizer/`

- Raw text is downloaded from a URL (a small corpus for learning).
- Text is tokenized using **Byte Pair Encoding (BPE)**.
- Token sequences are split into overlapping chunks using a sliding window.
- Each chunk forms:
  - **Input tokens**
  - **Target tokens** (shifted by one position)

This prepares the data for autoregressive training.

---

## Dataset & DataLoaders

**Location:** `src/data_pipeline/`

- `GPTDataset` creates `(input_ids, target_ids)` pairs.
- Separate **train** and **validation** splits are created.
- PyTorch `DataLoader` handles batching and shuffling.

This stage ensures the model sees text in fixed-length contexts, just like GPT.

---

## Model Architecture

**Location:** `src/model/`

The model is built step by step:

- Embedding layers (token + positional embeddings)
- Self-Attention
- Multi-Head Attention
- Feed-Forward Network
- Transformer Blocks
- Final Language Modeling Head

Each component is implemented explicitly to make the internals easy to understand.

---

## Text Generation

**Location:** `src/generate/`

- Implements autoregressive text generation.
- The model predicts one token at a time.
- Context window is maintained during generation.
- Used to sample text after each training epoch.

This helps visualize what the model is learning.

---

## Training & Evaluation

**Location:** `src/training/`

- Training loop with:
  - Forward pass
  - Loss computation
  - Backpropagation
  - Optimizer step
- Periodic evaluation on validation data.
- Text samples generated after each epoch.

The focus is correctness and clarity, not speed.

---

## Entry Scripts

**Location:** `scripts/`

- High-level scripts wire everything together:
  - Data loading
  - Model initialization
  - Training
  - Text generation

This separation keeps the core logic reusable and clean.

---

## How to Run

Make sure you are in the **project root directory**.

### Run Training

```bash
python -m scripts.train

---

## 🧠 How to Explore This Repository

1. `src/tokenizer/` → how text becomes tokens  
2. `src/data_pipeline/` → how training data is created  
3. `src/model/` → how GPT is built block by block  
4. `src/generate/` → how text generation works  
5. `src/training/` → how everything is trained together  
6. `scripts/train.py` → full pipeline in action  

Each section can be studied independently.
