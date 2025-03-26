import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import random
import os
import numpy as np
from collections import deque
import copy

# Create models directory
os.makedirs("models", exist_ok=True)

class Config:
    def __init__(self):
        self.vocab_size = None
        self.d_model = 256
        self.nhead = 16
        self.num_encoder_layers = 4
        self.num_decoder_layers = 4
        self.dim_feedforward = 1024
        self.dropout = 0.05

        self.num_epochs = 20
        self.batch_size = 32
        self.learning_rate = 0.001
        self.evolution_interval = 5
        self.population_size = 3
        self.mutation_rate = 0.01

        self.rlhf_epochs = 5
        self.rlhf_batch_size = 16
        self.reward_buffer_size = 1000
        self.entropy_coeff = 0.01
        self.value_coeff = 0.5
        self.gamma = 0.99
        self.lam = 0.95
        self.clip_param = 0.2
        self.ppo_epochs = 3
        self.ppo_batch_size = 8
        self.reward_model_learning_rate = 1e-4
        self.max_grad_norm = 1.0

        self.max_len = 50
        self.pad_token = "<pad>"
        self.start_token = "<s>"
        self.end_token = "</s>"

        try:
            import torch_xla.core.xla_model as xm
            self.device = xm.xla_device()
        except ImportError:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()

class TransformerChatModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.d_model = config.d_model
        self.src_embedding = nn.Embedding(vocab_size, self.d_model)
        self.tgt_embedding = nn.Embedding(vocab_size, self.d_model)
        self.transformer = nn.Transformer(
            self.d_model, config.nhead, config.num_encoder_layers,
            config.num_decoder_layers, config.dim_feedforward, config.dropout
        )
        self.fc_out = nn.Linear(self.d_model, vocab_size)

    def forward(self, src, tgt):
        src = self.src_embedding(src) * math.sqrt(self.d_model)
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        output = self.transformer(src.transpose(0,1), tgt.transpose(0,1))
        return self.fc_out(output.transpose(0,1))

class RewardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(config.d_model * 2, 512),  
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, src_embeddings, tgt_embeddings):
        return self.encoder(torch.cat([src_embeddings.mean(dim=1), tgt_embeddings.mean(dim=1)], dim=1))

def build_vocab(dataset_texts):
    tokens = {config.pad_token, config.start_token, config.end_token, "<sep>"}
    for text in dataset_texts:
        tokens.update(text.split())
    vocab = {word: idx for idx, word in enumerate(sorted(tokens))}
    return vocab, {idx: word for word, idx in vocab.items()}

def tokenize(text, vocab):
    return [vocab.get(token, vocab[config.pad_token]) for token in text.split()]

def parse_dataset(filename):
    with open(filename, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return [(lines[i][5:], lines[i+1][7:]) for i in range(0, len(lines), 2) if lines[i].startswith("USER:")]

pairs = parse_dataset("dataset.txt")
all_texts = [text for pair in pairs for text in pair]
vocab, inv_vocab = build_vocab(all_texts)
config.vocab_size = len(vocab)

train_data = []
for inp, resp in pairs:
    src_tokens = [vocab[config.start_token]] + tokenize(inp, vocab)[:config.max_len-2] + [vocab[config.end_token]]
    tgt_tokens = [vocab[config.start_token]] + tokenize(resp, vocab)[:config.max_len-2] + [vocab[config.end_token]]
    src_tokens += [vocab[config.pad_token]] * (config.max_len - len(src_tokens))
    tgt_tokens += [vocab[config.pad_token]] * (config.max_len - len(tgt_tokens))
    train_data.append((torch.tensor(src_tokens).unsqueeze(0), torch.tensor(tgt_tokens).unsqueeze(0)))

device = config.device
model = TransformerChatModel(config.vocab_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
reward_model = RewardModel().to(device)
reward_optimizer = optim.Adam(reward_model.parameters(), lr=config.reward_model_learning_rate)

# Train the model
for epoch in range(config.num_epochs):
    model.train()
    epoch_loss = 0
    for src, tgt in train_data:
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        tgt_input = tgt[:, :-1]
        output = model(src, tgt_input)
        loss = F.cross_entropy(output.view(-1, config.vocab_size), tgt[:, 1:].reshape(-1), ignore_index=vocab[config.pad_token])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(train_data):.4f}")

# Save the trained model
torch.save(model.state_dict(), "models/transformer_chat_model.pth")
torch.save(reward_model.state_dict(), "models/reward_model.pth")

# Save vocab
with open("models/vocab.txt", "w") as f:
    for word, idx in vocab.items():
        f.write(f"{word} {idx}\n")

print("Training complete. Model and vocab saved in 'models/' folder.")