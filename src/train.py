import torch
import torch.nn as nn
from torch.nn import functional as F
from .data import get_batch

@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, block_size, eval_iters, device):
    out = {}
    model.eval()
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, batch_size, block_size, device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def train_model(model, train_data, val_data, batch_size, block_size, max_iters, learning_rate, eval_interval, eval_iters, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    for i in range(max_iters):
        if i % eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, batch_size, block_size, eval_iters, device)
            print(f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch(train_data, batch_size, block_size, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    return model 