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

class Config:
    def __init__(self):
        # Model parameters
        self.vocab_size = None  # Determined by the data
        self.d_model = 256
        self.nhead = 16
        self.num_encoder_layers = 4
        self.num_decoder_layers = 4
        self.dim_feedforward = 1024
        self.dropout = 0.05

        # Training parameters
        self.num_epochs = 20
        self.batch_size = 32
        self.learning_rate = 0.001
        self.evolution_interval = 5
        self.population_size = 3
        self.mutation_rate = 0.01

        # RLHF parameters
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

        # Data parameters
        self.max_len = 50
        self.pad_token = "<pad>"
        self.start_token = "<s>"
        self.end_token = "</s>"

        # Device configuration
        self.use_tpu = False
        try:
            import torch_xla.core.xla_model as xm
            self.device = xm.xla_device()
            self.use_tpu = True
            print("Using TPU")
        except ImportError:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print("Using CUDA")
            else:
                self.device = torch.device("cpu")
                print("Using CPU")

config = Config()

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerChatModel(nn.Module):
    def __init__(self, vocab_size, d_model=None, nhead=None, num_encoder_layers=None,
                 num_decoder_layers=None, dim_feedforward=None, dropout=None):
        super(TransformerChatModel, self).__init__()
        self.d_model = d_model if d_model else config.d_model
        self.src_embedding = nn.Embedding(vocab_size, self.d_model)
        self.tgt_embedding = nn.Embedding(vocab_size, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model)
        self.transformer = nn.Transformer(
            self.d_model,
            nhead if nhead else config.nhead,
            num_encoder_layers if num_encoder_layers else config.num_encoder_layers,
            num_decoder_layers if num_decoder_layers else config.num_decoder_layers,
            dim_feedforward if dim_feedforward else config.dim_feedforward,
            dropout if dropout else config.dropout
        )
        self.fc_out = nn.Linear(self.d_model, vocab_size)
        self.value_head = nn.Linear(self.d_model, 1)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        src = self.src_embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        output = self.transformer(src.transpose(0,1), tgt.transpose(0,1),
                                  src_mask=src_mask, tgt_mask=tgt_mask, memory_mask=memory_mask)
        logits = self.fc_out(output.transpose(0,1))
        values = self.value_head(output.transpose(0,1))
        return logits, values

    def generate(self, src, max_len=None, start_symbol=None):
        self.eval()
        src = self.src_embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        memory = self.transformer.encoder(src.transpose(0,1))
        ys = torch.ones(1, 1).fill_(start_symbol).long().to(src.device)

        for i in range((max_len if max_len else config.max_len)-1):
            tgt = self.tgt_embedding(ys) * math.sqrt(self.d_model)
            tgt = self.pos_encoder(tgt)
            tgt_mask = self.transformer.generate_square_subsequent_mask(ys.size(0)).to(src.device)
            out = self.transformer.decoder(tgt.transpose(0,1), memory, tgt_mask=tgt_mask)
            out = self.fc_out(out.transpose(0,1))
            prob = out[-1, 0].softmax(dim=0)
            next_word = torch.multinomial(prob, 1)
            ys = torch.cat([ys, next_word.unsqueeze(0)], dim=0)
            if next_word.item() == vocab[config.end_token]:
                break
        return ys.squeeze().tolist()

class RewardModel(nn.Module):
    def __init__(self, d_model=None):
        super(RewardModel, self).__init__()
        self.d_model = d_model if d_model else config.d_model
        self.encoder = nn.Sequential(
            nn.Linear(self.d_model * 2, 512),  # Input is concatenated src+tgt embeddings
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, src_embeddings, tgt_embeddings):
        src_pooled = src_embeddings.mean(dim=1)
        tgt_pooled = tgt_embeddings.mean(dim=1)
        combined = torch.cat([src_pooled, tgt_pooled], dim=1)
        return self.encoder(combined)

def build_vocab(dataset_texts):
    tokens = set()
    tokens.update([config.pad_token, config.start_token, config.end_token, "<sep>"])
    for text in dataset_texts:
        tokens.update(text.split())
    vocab = {word: idx for idx, word in enumerate(sorted(tokens))}
    inv_vocab = {idx: word for word, idx in vocab.items()}
    return vocab, inv_vocab

def tokenize(text, vocab):
    return [vocab.get(token, vocab[config.pad_token]) for token in text.split()]

def detokenize(indices, inv_vocab):
    return " ".join([inv_vocab.get(idx, "<unk>") for idx in indices])

def parse_dataset(filename):
    with open(filename, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    pairs = []
    current_user = None
    for line in lines:
        if line.startswith("USER:"):
            current_user = line[len("USER:"):].strip()
        elif line.startswith("MUFFIN:") and current_user is not None:
            response = line[len("MUFFIN:"):].strip()
            pairs.append((current_user, response))
            current_user = None
    return pairs

def prepare_data(pairs, vocab, max_len=None):
    max_len = max_len if max_len else config.max_len
    data = []
    for inp, resp in pairs:
        # Tokenize and truncate before adding special tokens
        inp_tokens = tokenize(inp, vocab)[:max_len-2]
        resp_tokens = tokenize(resp, vocab)[:max_len-2]

        # Add special tokens
        src_tokens = [vocab[config.start_token]] + inp_tokens + [vocab[config.end_token]]
        tgt_tokens = [vocab[config.start_token]] + resp_tokens + [vocab[config.end_token]]

        # Pad sequences
        src_tokens = src_tokens + [vocab[config.pad_token]]*(max_len - len(src_tokens))
        tgt_tokens = tgt_tokens + [vocab[config.pad_token]]*(max_len - len(tgt_tokens))

        data.append((torch.tensor(src_tokens).unsqueeze(0), torch.tensor(tgt_tokens).unsqueeze(0)))
    return data

class RewardBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

    def add(self, src, tgt, reward):
        self.buffer.append((src, tgt, reward))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)

def get_reward(src_embeddings, tgt_embeddings, human_feedback=None):
    if human_feedback is not None:
        return torch.tensor(human_feedback, dtype=torch.float32).to(config.device)
    with torch.no_grad():
        return reward_model(src_embeddings, tgt_embeddings)

def train_reward_model(batch_size):
    if len(reward_buffer) < batch_size:
        return None

    batch = reward_buffer.sample(batch_size)
    src_embeddings = torch.cat([item[0] for item in batch])
    tgt_embeddings = torch.cat([item[1] for item in batch])
    rewards = torch.tensor([item[2] for item in batch], dtype=torch.float32).to(config.device)

    reward_optimizer.zero_grad()
    pred_rewards = reward_model(src_embeddings, tgt_embeddings).squeeze()
    loss = F.mse_loss(pred_rewards, rewards)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(reward_model.parameters(), config.max_grad_norm)
    reward_optimizer.step()

    return loss.item()

def compute_advantages(rewards, values, gamma=0.99, lam=0.95):
    advantages = torch.zeros_like(rewards)
    last_advantage = 0

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * (values[t+1] if t+1 < len(values) else 0) - values[t]
        advantages[t] = last_advantage = delta + gamma * lam * last_advantage

    return advantages

def ppo_update(model, batch, clip_param=0.2):
    src_tensors = torch.cat([item[0] for item in batch])
    tgt_tensors = torch.cat([item[1] for item in batch])
    old_log_probs = torch.cat([item[2] for item in batch])
    old_values = torch.cat([item[3] for item in batch])
    rewards = torch.cat([item[4] for item in batch])
    advantages = torch.cat([item[5] for item in batch])

    logits, values = model(src_tensors, tgt_tensors[:, :-1])
    log_probs = F.log_softmax(logits, dim=-1)
    new_log_probs = log_probs.gather(-1, tgt_tensors[:, 1:].unsqueeze(-1)).squeeze(-1)

    ratio = (new_log_probs - old_log_probs).exp()
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    value_loss = F.mse_loss(values.squeeze(-1), rewards)
    entropy = -(log_probs * log_probs.exp()).mean()

    loss = policy_loss + config.value_coeff * value_loss - config.entropy_coeff * entropy

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
    optimizer.step()

    return loss.item()

def evaluate_model(model, data):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt in data:
            src = src.to(config.device)
            tgt = tgt.to(config.device)
            tgt_input = tgt[:, :-1]
            targets = tgt[:, 1:]
            tgt_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(config.device)
            output, _ = model(src, tgt_input, tgt_mask=tgt_mask)
            loss = F.cross_entropy(output.reshape(-1, config.vocab_size), targets.reshape(-1),
                                 ignore_index=vocab[config.pad_token])
            total_loss += loss.item()
    model.train()
    return total_loss / len(data)

def mutate_model(model, mutation_rate=None):
    mutation_rate = mutation_rate if mutation_rate else config.mutation_rate
    mutated_model = TransformerChatModel(config.vocab_size).to(config.device)
    mutated_model.load_state_dict(model.state_dict())
    with torch.no_grad():
        for param in mutated_model.parameters():
            noise = torch.randn_like(param) * mutation_rate * param.std()
            param.add_(noise)
    return mutated_model

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0,1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# Initialize models and data
pairs = parse_dataset("dataset.txt")
all_texts = [text for pair in pairs for text in pair]
vocab, inv_vocab = build_vocab(all_texts)
config.vocab_size = len(vocab)
train_data = prepare_data(pairs, vocab)

device = config.device
model = TransformerChatModel(config.vocab_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
reward_model = RewardModel().to(device)
reward_optimizer = optim.Adam(reward_model.parameters(), lr=config.reward_model_learning_rate)
reward_buffer = RewardBuffer(config.reward_buffer_size)

# Initial supervised training
for epoch in range(1, config.num_epochs+1):
    model.train()
    epoch_loss = 0
    for src, tgt in train_data:
        src = src.to(device)
        tgt = tgt.to(device)
        optimizer.zero_grad()

        tgt_input = tgt[:, :-1]
        targets = tgt[:, 1:]
        tgt_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(device)

        output, _ = model(src, tgt_input, tgt_mask=tgt_mask)
        loss = F.cross_entropy(output.reshape(-1, config.vocab_size), targets.reshape(-1),
                             ignore_index=vocab[config.pad_token])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

        if config.use_tpu:
            import torch_xla.core.xla_model as xm
            xm.optimizer_step(optimizer)
        else:
            optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch}, Loss: {epoch_loss/len(train_data):.4f}")

    if epoch % config.evolution_interval == 0:
        print("Evolutionary update...")
        best_model = model
        best_loss = evaluate_model(model, train_data)
        for i in range(config.population_size):
            mutated_model = mutate_model(model)
            mutated_loss = evaluate_model(mutated_model, train_data)
            print(f"  Mutant {i+1} Loss: {mutated_loss:.4f}")
            if mutated_loss < best_loss:
                best_loss = mutated_loss
                best_model = mutated_model
        model.load_state_dict(best_model.state_dict())

# RLHF training
for rlhf_epoch in range(config.rlhf_epochs):
    model.eval()
    trajectories = []
    indices = torch.randperm(len(train_data))[:config.rlhf_batch_size]

    with torch.no_grad():
        for idx in indices:
            src, _ = train_data[idx]
            src = src.to(device)

            tgt_tokens = model.generate(src, max_len=config.max_len, start_symbol=vocab[config.start_token])
            tgt_tensor = torch.tensor(tgt_tokens).unsqueeze(0).to(device)

            tgt_input = tgt_tensor[:, :-1]
            logits, values = model(src, tgt_input)
            log_probs = F.log_softmax(logits, dim=-1)
            action_log_probs = log_probs.gather(-1, tgt_tensor[:, 1:].unsqueeze(-1)).squeeze(-1)

            src_embeddings = model.src_embedding(src) * math.sqrt(model.d_model)
            src_embeddings = model.pos_encoder(src_embeddings)
            tgt_embeddings = model.tgt_embedding(tgt_tensor) * math.sqrt(model.d_model)
            tgt_embeddings = model.pos_encoder(tgt_embeddings)

            reward = get_reward(src_embeddings, tgt_embeddings, human_feedback=0.8)  # Simulated feedback
            reward_buffer.add(src_embeddings, tgt_embeddings, reward.item())

            rewards = torch.ones_like(action_log_probs) * reward
            advantages = compute_advantages(rewards, values.squeeze(-1))

            trajectories.append((src, tgt_tensor, action_log_probs, values.squeeze(-1), rewards, advantages))

    # Train reward model
    if len(reward_buffer) >= config.rlhf_batch_size:
        reward_loss = train_reward_model(config.rlhf_batch_size)
        print(f"RLHF Epoch {rlhf_epoch+1}, Reward Loss: {reward_loss:.4f}")

    # PPO updates
    model.train()
    for ppo_epoch in range(config.ppo_epochs):
        random.shuffle(trajectories)
        for i in range(0, len(trajectories), config.ppo_batch_size):
            batch = trajectories[i:i+config.ppo_batch_size]
            loss, _, _, _ = ppo_update(model, batch)
            print(f"  PPO Epoch {ppo_epoch+1}, Batch {i//config.ppo_batch_size+1}, Loss: {loss:.4f}")

def chat(model, vocab, inv_vocab, device, max_len=None):
    model.eval()
    print("\nChat with MUFFIN (type 'quit' to exit):")
    while True:
        user_input = input("USER: ")
        if user_input.lower() == 'quit':
            break

        src_tokens = [vocab[config.start_token]] + tokenize(user_input, vocab)
        src_tokens = src_tokens[:max_len if max_len else config.max_len-1] + [vocab[config.end_token]]
        src_tokens = src_tokens + [vocab[config.pad_token]]*(max_len if max_len else config.max_len - len(src_tokens))

        src_tensor = torch.tensor(src_tokens).unsqueeze(0).to(device)
        response_tokens = model.generate(src_tensor, max_len=max_len if max_len else config.max_len,
                                       start_symbol=vocab[config.start_token])

        response_words = []
        for token in response_tokens:
            if token == vocab[config.end_token]:
                break
            response_words.append(inv_vocab.get(token, "<unk>"))
        print("MUFFIN:", " ".join(response_words))

chat(model, vocab, inv_vocab, device)
