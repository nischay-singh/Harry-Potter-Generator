import torch

def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text

def create_vocab(text):
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    return stoi, itos, vocab_size

def encode(text, stoi):
    return [stoi[c] for c in text]

def decode(tokens, itos):
    return "".join([itos[i] for i in tokens])

def get_batch(data, batch_size, block_size, device):
    idx = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i: i + block_size] for i in idx])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in idx])
    x, y = x.to(device), y.to(device)
    return x, y

def prepare_data(text, train_ratio=0.8):
    stoi, itos, vocab_size = create_vocab(text)
    data = torch.tensor(encode(text, stoi), dtype=torch.long)
    n = int(train_ratio * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data, stoi, itos, vocab_size 