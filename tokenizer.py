class SimpleTokenizerV1:
  def __init__(self, vocab):
    self.str_to_int = vocab    ##vocab is mapping of str to int so str_to_int is vocab directly
    self.int_to_str = {i: s for s,i in vocab.items()}

  def encode(self, text):
    preprocessed = re.split(r'([,.:;?_!()\']|--|\s)', text) #split the text

    preprocessed = [
        item.strip() for item in preprocessed if item.strip()  ## removing white spaces
    ]

    ids = [self.str_to_int[s] for s in preprocessed]
    return ids

  def decode(self, ids):
    text = " ".join([self.int_to_str[i] for i in ids])  #get the tokens from ids and join those tokens
    # Replace spaces which are present before specified punctuations
    text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
    return text


tokenizer = SimpleTokenizerV1(vocab)

# testing the encode method by passing the sample text
# text is from "training set"
text = """"It's the last he painted, you know,"
           Mrs. Gisburn said with pardonable pride."""

ids = tokenizer.encode(text)
print(ids)

# test the decode method
text = tokenizer.decode(ids)
print(text)

