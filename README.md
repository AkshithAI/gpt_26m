# GPT-26M: A 26 Million Parameter GPT Model

A lightweight GPT implementation trained on the TinyStories dataset with ~26 million parameters. This project includes custom attention mechanisms, RoPE (Rotary Position Embeddings), and optional Mixture of Experts (MoE) support.

## 📁 Project Structure

```
gpt_26m/
├── src/                    # Core model implementation
│   ├── gpt.py             # Main GPT model architecture
│   ├── attention.py       # Multi-head attention implementation
│   ├── moe.py             # Mixture of Experts layer
│   ├── rope.py            # Rotary Position Embeddings
│   ├── config.py          # Model configuration
│   ├── tokenizer.py       # Tokenizer setup
│   └── dataloader.py      # Dataset loading and preprocessing
├── scripts/               # Training and inference scripts
│   ├── train.py          # Training script with wandb logging
│   └── test_generate.py  # Text generation script
├── notebooks/            # Jupyter notebooks for experiments
├── assets/              # Model weights and tokenizer files
│   ├── 3hr_gpt_model.pth
│   └── tinystories_tokenizer1.json
├── checkpoints/         # Training checkpoints (gitignored)
├── data/               # Dataset cache (gitignored)
└── requirements.txt    # Python dependencies
```

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/AkshithAI/gpt_26m.git
cd gpt_26m
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Training

Train the model from scratch:
```bash
python scripts/train.py
```

Training features:
- Mixed precision training (FP16)
- Gradient clipping
- Cosine learning rate schedule with warmup
- Early stopping
- WandB integration for experiment tracking
- Automatic checkpointing and artifact logging

### Text Generation

Generate text using a trained model:
```bash
python scripts/test_generate.py
```

## 🏗️ Model Architecture

- **Parameters**: ~26 million
- **Embedding Dimension**: Configurable (see `src/config.py`)
- **Attention Heads**: Multi-head self-attention
- **Position Encoding**: RoPE (Rotary Position Embeddings)
- **Context Length**: Configurable
- **Optional**: Mixture of Experts (MoE) layers

## 📊 Training Configuration

Key hyperparameters (configurable in `src/config.py`):
- Learning rate with cosine scheduling
- Warmup steps: 5000
- Optimizer: AdamW with weight decay
- Mixed precision training (FP16)
- Gradient clipping: max norm 1.0

## 📈 Monitoring

The training script logs metrics to Weights & Biases:
- Training loss (per batch and per epoch)
- Validation loss
- Learning rate
- Gradient norms
- Model checkpoints as artifacts

## 🎯 Dataset

Trained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset - a collection of short stories generated for training small language models.

## 📝 License

MIT License

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on GitHub.
