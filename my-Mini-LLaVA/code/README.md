# Mini Vision Language Model (VLM) Project

This repository contains a minimalistic implementation of a Vision Language Model (VLM) that integrates a pre-trained vision encoder with a powerful Large Language Model (LLM) via a custom mapping network. The project follows a two-stage training approach similar to modern VLMs like LLaVA.


## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

- `model.py`: Contains the core architecture of the VLM, including the Vision Encoder, Mapping Network, and LLM integration
- `data_preprocess.py`: Unified data preprocessing for both training stages
- `train.py`: Unified training script supporting both alignment and instruction stages
- `interactive_inference.ipynb`: Jupyter notebook for interactive model testing
- `requirements.txt`: Required dependencies
- `scripts/`: Directory containing shell scripts for different model configurations

## Core Architecture

- **Vision Encoder**: Pre-trained OpenAI CLIP ViT-L/14 model
- **Large Language Model (LLM)**: Qwen2-1.5B-Instruct model
- **Mapping Network**: A Multi-Layer Perceptron (MLP) implemented from scratch in PyTorch
- **VLM Assembly**: A custom `torch.nn.Module` that combines the three components


