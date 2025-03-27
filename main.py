from datasets import Dataset
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the dataset from a text file.
    
    Args:
        file_path (str): Path to the input dataset file
    
    Returns:
        datasets.Dataset: Preprocessed dataset
    """
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        # Preprocess lines
        data = []
        for line in lines:
            # Split by tab and strip whitespace
            parts = line.strip().split('\t')
            data.append({
                'input': parts[0],
                'response': parts[1]
            })

        # Convert to pandas DataFrame
        df = pd.DataFrame(data)

        # Convert to Hugging Face Dataset
        return Dataset.from_pandas(df)

    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def tokenize_function(tokenizer, examples):
    """
    Tokenize input and response pairs.
    
    Args:
        tokenizer: Tokenizer to use
        examples (dict): Dictionary of input examples
    
    Returns:
        dict: Tokenized inputs
    """
    # Combine input and response for training
    combined_texts = [f"{input} {response}" for input, response in zip(examples['input'], examples['response'])]
    
    return tokenizer(
        combined_texts, 
        truncation=True, 
        padding='max_length', 
        max_length=512, 
        return_tensors='pt'
    )

def main():
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load and preprocess the dataset
    dataset = load_and_preprocess_data('dataset3.txt')
    
    if dataset is None:
        print("Failed to load dataset. Exiting.")
        return

    # 2. Load tokenizer and model
    model_name = 'FlameF0X/Muffin-2.9a-0C17'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add padding token if not existing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    # 3. Tokenize the dataset
    tokenized_datasets = dataset.map(
        lambda examples: tokenize_function(tokenizer, examples), 
        batched=True, 
        remove_columns=dataset.column_names
    )

    # 4. Prepare data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False  # Not using masked language modeling
    )

    # 5. Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy='steps',  # Add evaluation strategy
        save_strategy='steps',  # Add save strategy
        save_steps=500,  # Save checkpoint every 500 steps
        learning_rate=5e-5,  # Add learning rate
        fp16=torch.cuda.is_available(),  # Enable mixed precision if GPU available
    )

    # 6. Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        data_collator=data_collator,
    )

    # 7. Train the model
    trainer.train()

    # 8. Save the fine-tuned model
    trainer.save_model('./fine_tuned_muffin')
    tokenizer.save_pretrained('./fine_tuned_muffin')
    print("Model training and saving completed successfully!")

if __name__ == '__main__':
    main()
