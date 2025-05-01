import torch
from .data import decode

def generate_text(model, stoi, itos, device, block_size, max_new_tokens=100):
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_tokens = model.generate(context, max_new_tokens, block_size)[0].tolist()
    return decode(generated_tokens, itos) 