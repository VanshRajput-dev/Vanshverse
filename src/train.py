import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TextDataset,
)
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
from tqdm import tqdm

def load_dataset(tokenizer, file_path="data/cleaned_poems.txt", block_size=BLOCK_SIZE):
    print(f"📖 Loading dataset from {file_path} ...")
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size
    )
    print(f"📚 Loaded {len(dataset)} text chunks for training.")
    return dataset

def train_model():
    print("🚀 Initializing tokenizer and model...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

    tokenizer.add_special_tokens({
        "additional_special_tokens": ["<|startofpoem|>", "<|endofpoem|>"]
    })
    model.resize_token_embeddings(len(tokenizer))

    train_dataset = load_dataset(tokenizer)

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
        report_to=None
    )

    print("🧠 Starting training ...")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    total_steps = int(len(train_dataset) / BATCH_SIZE * EPOCHS)
    print(f"🕒 Estimated total steps: {total_steps}")

    # Wrap the Trainer's train() with tqdm for visual feedback
    with tqdm(total=total_steps, desc="🚀 Training Progress", unit="step") as pbar:
        for epoch in range(EPOCHS):
            print(f"\n🌙 Epoch {epoch + 1}/{EPOCHS}")
            trainer.train()
            pbar.update(int(len(train_dataset) / BATCH_SIZE))

    print("💾 Saving model and tokenizer...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    train_model()
