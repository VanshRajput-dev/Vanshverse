import os
import sys

# Fix imports for project structure
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["TORCHDYNAMO_DISABLE"] = "1"

from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from utils.config import (
    MODEL_NAME,
    OUTPUT_DIR,
    BLOCK_SIZE,
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    SAVE_STEPS,
    LOGGING_STEPS,
)

def load_poems_dataset(tokenizer, file_path="data/cleaned_poems.txt", block_size=BLOCK_SIZE):
    print(f"Loading dataset from {file_path}...")
    dataset = load_dataset("text", data_files={"train": file_path})
    print(f"Loaded {len(dataset['train'])} poems for training.")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=block_size,
        )

    print("Tokenizing dataset...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    return tokenized_datasets["train"]

def train_model():
    print("Initializing tokenizer and model...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

    # Add special tokens for poem structure
    tokenizer.add_special_tokens({
        "additional_special_tokens": ["<|startofpoem|>", "<|endofpoem|>"]
    })

    # Add a pad token (GPT-2 does not have one by default)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.config.pad_token_id = tokenizer.pad_token_id

    model.resize_token_embeddings(len(tokenizer))

    train_dataset = load_poems_dataset(tokenizer)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        logging_steps=LOGGING_STEPS,
        prediction_loss_only=True,
    )

    print("Starting fine-tuning...")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("Training interrupted manually. Saving progress...")

    print("Saving model and tokenizer...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved successfully to {OUTPUT_DIR}")

if __name__ == "__main__":
    train_model()
