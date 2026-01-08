import tiktoken

def my_tokenizer():
    """
    Returns GPT-2 Byte Pair Encoding(BPE) tokenizer from tiktoken
    
    """
    return tiktoken.get_encoding("gpt2")
    

