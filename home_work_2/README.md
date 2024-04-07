# Model GPT based on a Transformer.
gpt.py - modified model with Evolved Transformer, activation function GeLU and hyperparameters (18 layers, dff = 2048, H = 8).

![Perplexity](perplexity.png)

* Base - Base model with activation function ReLU and hyperparameters (6 layers, dff = 1536, H = 6);
* Modified - Model Transformer with activation function GeLU and hyperparameters (18 layers, dff = 2048, H = 8);
* Switch - Switch Transformer with activation function GeLU and hyperparameters (18 layers, dff = 2048, H = 8);
* Lightweight - Lightweight convolution Transformer with activation function GeLU and hyperparameters (18 layers, dff = 2048, H = 8);
* Evolved - Evolved Transformer with activation function GeLU and hyperparameters (18 layers, dff = 2048, H = 8).
