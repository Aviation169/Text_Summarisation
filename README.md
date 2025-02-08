Transformer-based Text Summarization(My own designed transformer)
[!This project is just for my practice for real-time use; it required high-level training]

Overview:
This repository contains a transformer-based model for abstractive text summarization. The model is built using PyTorch and leverages the BERT tokenizer and embeddings for tokenization and feature extraction. The architecture consists of transformer-based encoder-decoder layers for generating summaries from input articles.

Project Structure:
PositionalEncoding: Implements positional encoding to capture the position of tokens in the input sequence.
TransformerEncoderLayer: Implements a single layer of the encoder using multi-head self-attention and feed-forward networks.
TransformerDecoderLayer: Implements a single layer of the decoder using multi-head self-attention and cross-attention with the encoder.
TransformerSummarizer: The core model combining the encoder and decoder, responsible for generating the final summary.
SummarizationDataset: A custom dataset class for loading and preprocessing the data (articles and summaries).
Main Script: Contains the logic for model initialization, training, and evaluation.

Dependencies:
PyTorch
Transformers (Hugging Face)
tqdm
pandas

Uses:
Pre-trained BERT Model: The code uses the bert-base-uncased model from Hugging Face's Transformers library for tokenization and embedding generation.
Training: The model is trained on a dataset that contains articles and their corresponding highlights. The dataset should be in CSV format with columns article and highlights.
Model Initialization: The TransformerSummarizer class initializes the model with a BERT tokenizer and embeddings. You can customize the model parameters like the number of encoder and decoder layers, hidden dimensions, and dropout rates.
Training Loop: The model is trained using cross-entropy loss and optimized using Adam. It supports batch processing for training and validation.

Data Format:
The dataset should be a CSV file containing two columns:
article: The full article (text).
highlights: The summary of the article (text).
Example:
article,highlights
"Text of the article goes here.","Summary of the article."
"Another article text.","Summary of the second article."
Training and Evaluation
Run the main() function to start training the model. You can adjust the number of epochs, batch size, and learning rate as needed.

Hyperparameters:
d_model: The embedding dimension for the model. Default is 384.
nhead: Number of attention heads. Default is 4.
d_ff: Feed-forward layer dimension. Default is 1024.
num_encoder_layers: Number of encoder layers in the transformer. Default is 3.
num_decoder_layers: Number of decoder layers in the transformer. Default is 3.
max_seq_length: Maximum sequence length for both the source and target sequences. Default is 256.

Performance:
The model can be evaluated on a validation dataset after each epoch. You can monitor the training and validation loss to evaluate the model's performance.
