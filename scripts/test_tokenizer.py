from src.tokenizer.bpe_tokenizer import my_tokenizer

tokenizer = my_tokenizer()

text = "Hello Manoj"
tokens = tokenizer.encode(text)

print("Tokens:", tokens)
print("Decoded:", tokenizer.decode(tokens))
print("Vocab size:", tokenizer.n_vocab)