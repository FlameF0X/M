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
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datetime import datetime

class Config:
    def __init__(self):
        # Model parameters
        self.model_name = "distilgpt2"
        self.max_len = 50

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

        # Special tokens
        self.pad_token = "<pad>"
        self.start_token = "<s>"
        self.end_token = "</s>"
        self.sep_token = "<sep>"

        # Output configuration
        self.output_dir = "generated_responses"
        self.save_interval = 5  # Save every N epochs

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

class GPT2ChatModel(nn.Module):
    def __init__(self):
        super(GPT2ChatModel, self).__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained(config.model_name)
        self.value_head = nn.Linear(self.gpt2.config.n_embd, 1)
        
        # Add special tokens
        self.tokenizer = GPT2Tokenizer.from_pretrained(config.model_name)
        special_tokens = {
            'pad_token': config.pad_token,
            'sep_token': config.sep_token,
            'additional_special_tokens': [config.start_token, config.end_token]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        self.gpt2.resize_token_embeddings(len(self.tokenizer))

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        
        # Get the last hidden states for value prediction
        hidden_states = outputs.hidden_states[-1]
        values = self.value_head(hidden_states).squeeze(-1)
        
        return logits, values

    def generate(self, input_ids, attention_mask=None, max_length=None):
        return self.gpt2.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length if max_length else config.max_len,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )

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

def prepare_data(pairs, tokenizer, max_len=None):
    max_len = max_len if max_len else config.max_len
    data = []
    
    for inp, resp in pairs:
        # Format as "USER: <input> SEP MUFFIN: <response>"
        text = f"USER: {inp} {tokenizer.sep_token} MUFFIN: {resp}"
        
        # Tokenize
        encoding = tokenizer(
            text,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Create labels (mask out the user part for training)
        sep_pos = (encoding.input_ids[0] == tokenizer.sep_token_id).nonzero().item()
        labels = encoding.input_ids.clone()
        labels[:, :sep_pos+1] = -100  # ignore user part in loss
        
        data.append({
            'input_ids': encoding.input_ids,
            'attention_mask': encoding.attention_mask,
            'labels': labels
        })
    
    return data

def save_generated_responses(model, data, epoch, output_dir):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"responses_epoch_{epoch}_{timestamp}.txt")
    
    with torch.no_grad(), open(filename, "w", encoding="utf-8") as f:
        for i, batch in enumerate(data[:10]):  # Save responses for first 10 examples
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            
            # Generate response
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=config.max_len
            )
            
            # Decode the input and output
            input_text = model.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            response = model.tokenizer.decode(generated[0], skip_special_tokens=True)
            
            # Write to file
            f.write(f"Example {i+1}:\n")
            f.write(f"Input: {input_text}\n")
            f.write(f"Generated Response: {response}\n")
            f.write("\n" + "="*80 + "\n\n")
    
    print(f"Saved generated responses to {filename}")

# Initialize models and data
model = GPT2ChatModel().to(config.device)
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

# Load and prepare data
pairs = parse_dataset("dataset.txt")
train_data = prepare_data(pairs, model.tokenizer)

# Training loop
for epoch in range(1, config.num_epochs+1):
    model.train()
    epoch_loss = 0
    
    for batch in train_data:
        input_ids = batch['input_ids'].to(config.device)
        attention_mask = batch['attention_mask'].to(config.device)
        labels = batch['labels'].to(config.device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, labels=labels)
        loss = F.cross_entropy(
            outputs[0].view(-1, outputs[0].size(-1)),
            labels.view(-1),
            ignore_index=-100
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        
        if config.use_tpu:
            import torch_xla.core.xla_model as xm
            xm.optimizer_step(optimizer)
        else:
            optimizer.step()
        
        epoch_loss += loss.item()
    
    print(f"Epoch {epoch}, Loss: {epoch_loss/len(train_data):.4f}")

    # Save generated responses at intervals
    if epoch % config.save_interval == 0 or epoch == config.num_epochs:
        save_generated_responses(model, train_data, epoch, config.output_dir)

    # Evolutionary update
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

def evaluate_model(model, data):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in data:
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels = batch['labels'].to(config.device)
            
            outputs = model(input_ids, attention_mask)
            logits = outputs[0]
            
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
            total_loss += loss.item()
    
    model.train()
    return total_loss / len(data)

def mutate_model(model, mutation_rate=None):
    mutation_rate = mutation_rate if mutation_rate else config.mutation_rate
    mutated_model = GPT2ChatModel().to(config.device)
    mutated_model.load_state_dict(model.state_dict())
    
    with torch.no_grad():
        for param in mutated_model.parameters():
            if param.requires_grad:  # Only mutate trainable parameters
                noise = torch.randn_like(param) * mutation_rate * param.std()
                param.add_(noise)
    
    return mutated_model
