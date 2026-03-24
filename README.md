# GPT-2 From Scratch in PyTorch 🧠

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c)
![License](https://img.shields.io/badge/License-MIT-green)

A complete, end-to-end implementation of the GPT-2 large language model built entirely from scratch using PyTorch. 

This repository contains everything needed to understand the LLM lifecycle: custom tokenization, building fundamental attention mechanisms, pretraining on raw text, mapping official OpenAI weights, and fine-tuning the model for both text classification and instruction-following.

## 📑 Table of Contents
- [Architecture & Design](#-architecture--design)
- [Model Parameters](#-model-parameters)
- [Installation](#-installation)
- [How to Run](#-how-to-run)
  - [1. Pretraining (Next-Word Prediction)](#1-pretraining-next-word-prediction)
  - [2. Loading Pre-trained Weights](#2-loading-pre-trained-weights)
  - [3. Fine-Tuning: Spam Classification](#3-fine-tuning-spam-classification)
  - [4. Fine-Tuning: Instruction Following](#4-fine-tuning-instruction-following)
- [Automated Evaluation](#-automated-evaluation-via-ollama)
- [Acknowledgements](#-acknowledgements)

---

## 🏗️ Architecture & Design

This model implements a **decoder-only transformer architecture** that identically matches the official OpenAI GPT-2 specifications. 

### Core Components
* **Tokenization:** Uses OpenAI's BPE `tiktoken` alongside custom dataloaders that slice text into sliding context windows.
* **Embeddings:** Combines Token Embeddings and Positional Embeddings to retain sequence order.
* **Causal Multi-Head Attention:** Implements scaled dot-product attention with a causal mask to prevent the model from "looking ahead" at future tokens.
* **Transformer Blocks:** Combines Multi-Head Attention, Layer Normalization (`LayerNorm`), and a Feed-Forward Neural Network featuring `GELU` activations.

### Architecture Diagram

graph TD
    A[Input Text] -->|BPE Tokenizer| B[Token IDs]
    B --> C[Token Embedding]
    B --> D[Positional Embedding]
    C --> E((+))
    D --> E
    
    subgraph Transformer Block x N Layers
        E --> F[Layer Norm]
        F --> G[Multi-Head Causal Attention]
        G --> H((+ Residual))
        H --> I[Layer Norm]
        I --> J[Feed-Forward Network GELU]
        J --> K((+ Residual))
    end
    
    E -->|Input| F
    K --> L[Final Layer Norm]
    L --> M[Linear Output Head]
    M --> N[Logits Vocabulary Size]

Model Parameters
The architecture is highly modular. By adjusting the configuration dictionary, you can instantiate any standard GPT-2 size model. The default is set to GPT-2 Small (124M).

python
GPT_CONFIG_124M = {
    "vocab_size": 50257,     # Standard GPT-2 vocabulary size
    "context_length": 256,   # Context window (can scale to 1024)
    "emb_dim": 768,          # Dimensionality of embeddings
    "n_heads": 12,           # Number of attention heads
    "n_layers": 12,          # Number of transformer blocks
    "drop_rate": 0.1,        # Dropout rate for regularization
    "qkv_bias": False        # Query-Key-Value bias
}
Note: The code logic easily scales up to gpt2-medium (355M), gpt2-large (774M), and gpt2-xl (1558M) by changing emb_dim, n_heads, and n_layers.

💻 Installation
Clone the repository and install the required dependencies. It is recommended to use a virtual environment.

bash
git clone https://github.com/yourusername/gpt2-from-scratch.git
cd gpt2-from-scratch

# Install dependencies
pip install torch pandas matplotlib tiktoken tqdm numpy
🚀 How to Run
1. Pretraining (Next-Word Prediction)
To train the model from scratch on a provided text file (the-verdict.txt):

The script initializes the model and PyTorch DataLoader (using a sliding context window).

It optimizes using Cross-Entropy Loss and the AdamW optimizer.

Training plots (Loss vs. Epochs/Tokens) will be generated automatically via matplotlib.

2. Loading Pre-trained Weights
If you want to skip pretraining, the repository includes a script to download the official GPT-2 weights from OpenAI and map them directly into our custom PyTorch architecture.

Use download_and_load_gpt2() to pull the weights.

Use load_weights_into_gpt() to map them.

Test generation using the custom generate() function, which supports temperature and top_k sampling.

3. Fine-Tuning: Spam Classification
The model can be fine-tuned for sequence classification (e.g., detecting Spam vs. Ham SMS messages).

Automatically downloads the UCI SMS Spam Collection dataset.

Swaps the final linear output layer to output exactly 2 classes (model.out_head = torch.nn.Linear(in_features=768, out_features=2)).

Freezes core transformer layers and trains the classification head using standard cross-entropy.

4. Fine-Tuning: Instruction Following
The script supports instruction-based fine-tuning using a JSON dataset (instruction-data.json).

Automates formatting into a standard prompt template ("Below is an instruction that describes a task...").

Uses a custom collate_fn for dynamic padding (pad_token_id=50256).

Masks the prompt using ignore_index=-100 so that the model only calculates loss on the generated response.

📊 Automated Evaluation via Ollama
To evaluate the quality of the instruction-tuned model, this repository includes an automated evaluation script that uses a strong local LLM as a judge.

Install Ollama on your machine.

Pull the Llama 3 model:

bash
ollama pull llama3
Run the evaluation script:

bash
python ollama_evaluate.py --filepath instruction-data-with-response.json
This will score the model's generated responses based on accuracy and coherence, outputting an average evaluation metric.

🙏 Acknowledgements
This project was built following the curriculum of Sebastian Raschka's outstanding guide: Build a Large Language Model (From Scratch). Huge thanks to the author for making the concepts behind LLMs accessible and practical.
