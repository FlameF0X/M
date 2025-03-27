from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling

# 1. Load the dataset
with open('dataset3.txt', 'r') as file:
    lines = file.readlines()

# Remove empty lines
lines = [line for line in lines if line.strip() != '']

# Prepare lists for inputs (user messages) and responses (MUFFIN's replies)
inputs = []
responses = []

# Process each conversation pair
for i in range(0, len(lines), 2):  # Since each conversation consists of two lines (USER and MUFFIN)
    user_input = lines[i].strip().replace('USER:', '').strip()  # Get user input and remove the 'USER:' label
    muffin_response = lines[i + 1].strip().replace('MUFFIN:', '').strip()  # Get MUFFIN's response and remove the 'MUFFIN:' label
    
    inputs.append(user_input)
    responses.append(muffin_response)

# Convert the lists into a pandas DataFrame
df = pd.DataFrame({'input': inputs, 'response': responses})

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# 2. Tokenize the data
tokenizer = AutoTokenizer.from_pretrained('FlameF0X/Muffin-2.9a-0C17')

# Add padding token if not exists
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(
        examples['input'], 
        examples['response'], 
        truncation=True, 
        padding='max_length', 
        max_length=512
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 3. Load the pre-trained model
model = AutoModelForCausalLM.from_pretrained('FlameF0X/Muffin-2.9a-0C17')

# 4. Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# 5. Initialize the Trainer
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator,
)

# 6. Train the model
trainer.train()

# 7. Save the fine-tuned model
model.save_pretrained('./fine_tuned_muffin')
tokenizer.save_pretrained('./fine_tuned_muffin')