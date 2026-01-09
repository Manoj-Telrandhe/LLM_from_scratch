import torch
from torch.utils.data import DataLoader
from src.data_pipeline.dataset import GPTDataset
from src.tokenizer.bpe_tokenizer import my_tokenizer

def create_dataloader(
    txt,
    tokenizer,
    batch_size,
    max_length,
    stride,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    

    dataset = GPTDataset(txt, tokenizer, max_length, stride)


    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return dataloader
