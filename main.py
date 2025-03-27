from datasets import Dataset
import pandas as pd
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
)
import os

def load_and_preprocess_dataset(file_path):
    """
    Load dataset from file and preprocess it.
    
    Args:
        file_path (str): Path to the input text file
    
    Returns:
        pd.DataFrame: Processed DataFrame with inputs and responses
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
    except IOError as e:
        print(f"Error reading file: {e}")
        return None
    
    # Remove empty lines
    lines = [line.strip() for line in lines if line.strip()]
    
    # Prepare lists for inputs (user messages) and responses (MUFFIN's replies)
    inputs = []
    responses = []
    
    # Process each conversation pair
    try:
        for i in range(0, len(lines), 2):  # Since each conversation consists of two lines (USER and MUFFIN)
            user_input = lines[i].replace('USER:', '').strip()  # Get user input and remove the 'USER:' label
            muffin_response = lines[i + 1].replace('MUFFIN:', '').strip()  # Get MUFFIN's response and remove the 'MUFFIN:' label
            
            inputs.append(user_input)
            responses.append(muffin_response)
    except IndexError:
        print("Error: The dataset file might not have an even number of lines or is improperly formatted.")
        return None
    
    # Convert the lists into a pandas DataFrame
    return pd.DataFrame({'input': inputs, 'response': responses})

def main():
    # 1. Load and preprocess the dataset
    dataset_path = 'dataset3.txt'
    df = load_and_preprocess_dataset(dataset_path)
    
    if df is None:
        print("Failed to load or process the dataset. Exiting.")
        return
    
    # Convert to Hugging Face Dataset
    dataset = Dataset.from_pandas(df)
    
    # 2. Prepare tokenizer
    model_name = 'FlameF0X/Muffin-2.9a-0C17'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add padding token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        """Tokenize input-response pairs"""
        return tokenizer(
            examples['input'], 
            examples['response'], 
            truncation=True, 
            padding='max_length', 
            max_length=512, 
            return_tensors='pt'
        )
    
    # Tokenize the dataset
    tokenized_datasets = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=dataset.column_names
    )
    
    # 3. Load the pre-trained model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # 4. Define training arguments
    output_dir = './results'
    os.makedirs(output_dir, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy='steps',  # Added evaluation strategy
        save_strategy='steps',        # Added save strategy
        save_steps=500,               # Save checkpoints periodically
        load_best_model_at_end=True,  # Load the best model at the end of training
    )
    
    # 5. Initialize the Data Collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False  # Causal language modeling (not masked)
    )
    
    # 6. Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        data_collator=data_collator,
    )
    
    # 7. Train the model
    try:
        trainer.train()
        
        # 8. Save the fine-tuned model and tokenizer
        save_path = './fine_tuned_muffin'
        os.makedirs(save_path, exist_ok=True)
        
        trainer.save_model(save_path)
        tokenizer.save_pretrained(save_path)
        
        print(f"Model successfully fine-tuned and saved to {save_path}")
    
    except Exception as e:
        print(f"An error occurred during training: {e}")

if __name__ == '__main__':
    main()
