# Harry Potter Language Model

A transformer-based language model trained on the Harry Potter book series to generate text in the style of J.K. Rowling's writing.

## Project Structure
```
.
├── src/
│   ├── model.py          # Transformer model architecture
│   ├── data.py           # Data loading and preprocessing
│   ├── train.py          # Training loop and utilities
│   └── generate.py       # Text generation utilities
├── data/
│   └── harry_potter.txt  # Training data
├── main.py              # Main script to run the model
└── README.md
```

## Features
- Transformer-based architecture
- Multi-head attention mechanism
- Positional embeddings
- Layer normalization
- Dropout for regularization

## Requirements
- Python 3.x
- PyTorch
- CUDA (optional, for GPU acceleration)

## Installation
1. Clone this repository
2. Install dependencies:
```bash
pip install torch
```

## Usage
1. Place your Harry Potter text file in the `data` directory
2. Run the training script:
```bash
python main.py
```

## Model Architecture
- Embedding dimension: 384
- Number of attention heads: 6
- Number of transformer layers: 6
- Block size: 128
- Batch size: 64

## Training Parameters
- Learning rate: 3e-5
- Maximum iterations: 1000
- Evaluation interval: 500
- Dropout: 0.2 