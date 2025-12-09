# IMPLEMENTING A DATA LOADER 

from torch.utils.data import Dataset, DataLoader

class GPTDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]



# data loader
def create_dataloader(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):

  # Initialize the tokenizer
  tokenizer = tiktoken.get_encoding("gpt2")

  # create dataset instance
  dataset = GPTDataset(txt, tokenizer, max_length, stride)
  
  # create dataloader
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

  return dataloader

# open the text data file
with open("/content/verdict_story.txt", "r") as file:
  raw_text = file.read()

# create instance with batch_size = 1 and stride = 1
dataloader = create_dataloader(
    raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
)

# Convert dataloader into a python iterator to fetch the next entry via Python's built in next() function
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)


# batch_size = 8 and max_length = 4 
dataloader = create_dataloader(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)
