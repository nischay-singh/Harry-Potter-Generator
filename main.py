import torch
from src.model import BigramLanguageModel
from src.data import load_data, prepare_data, get_batch
from src.train import train_model
from src.generate import generate_text

batch_size = 64
block_size = 128
max_iters = 1000
learning_rate = 3e-5
eval_iters = 100
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
eval_interval = 500

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
torch.manual_seed(40)

def main():
    text = load_data("data/harry_potter.txt")
    train_data, val_data, stoi, itos, vocab_size = prepare_data(text)
    
    model = BigramLanguageModel(
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        block_size=block_size,
        dropout=dropout
    )
    model = model.to(device)
    
    model = train_model(
        model=model,
        train_data=train_data,
        val_data=val_data,
        batch_size=batch_size,
        block_size=block_size,
        max_iters=max_iters,
        learning_rate=learning_rate,
        eval_interval=eval_interval,
        eval_iters=eval_iters,
        device=device
    )
    
    generated_text = generate_text(model, stoi, itos, device, block_size)
    print("\nGenerated Text:")
    print(generated_text)

if __name__ == "__main__":
    main()