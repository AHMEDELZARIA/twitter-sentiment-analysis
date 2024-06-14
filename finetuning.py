from transformers import (
  TFAutoModelForSequenceClassification,
  create_optimizer,
  DataCollatorWithPadding,
  AutoTokenizer
)
from datasets import load_dataset
import tensorflow as tf
import numpy as np
import shutil
import os

def fine_tune_model(checkpoint, output_dir):
    try:
        # Load the model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = TFAutoModelForSequenceClassification.from_pretrained(
            checkpoint,
            num_labels=3,
            id2label={
                0: 'NEGATIVE',
                1: 'NEUTRAL',
                2: 'POSITIVE'
            },
            label2id={
                'NEGATIVE': 0,
                'NEUTRAL': 1,
                'POSITIVE': 2
            }
        )

        # Load and preprocess the datasets
        datasets = load_dataset("cardiffnlp/tweet_sentiment_multilingual")
        def tokenize_function(example):
            return tokenizer(example['text'], truncation=True)
        tokenized_datasets = datasets.map(tokenize_function, batched=True)

        # Data collator batches and pads our samples based on tokenizer rules
        batch_size = 8
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

        # Create TensorFlow dataset splits
        tf_train_dataset = tokenized_datasets["train"].to_tf_dataset(
            columns=["attention_mask", "input_ids"],
            label_cols=["label"],
            shuffle=True,
            collate_fn=data_collator,
            batch_size=batch_size,
        )

        tf_validation_dataset = tokenized_datasets["validation"].to_tf_dataset(
            columns=["attention_mask", "input_ids"],
            label_cols=["label"],
            shuffle=False,
            collate_fn=data_collator,
            batch_size=batch_size,
        )

        # Set-up optimizer
        num_epochs = 3
        num_train_steps = len(tf_train_dataset) * num_epochs
        optimizer, lr_scheduler = create_optimizer(
            init_lr=5e-5,
            num_warmup_steps=0,
            num_train_steps=num_train_steps,
            weight_decay_rate=0.01
        )

        # Compile and train the model
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"]
        )
        model.fit(
            tf_train_dataset,
            validation_data=tf_validation_dataset,
            epochs=num_epochs
        )

        # Save the model to google drive
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Indicate successful fine-tuning
        return True
    except Exception as e:
        print(f"Error during fine-tuning: {e}")
        return False # Indicate failed fine-tuning


def main():
    print("Model Finetuning for Sentiment Analysis on the Cardiff Twitter Sentiment Dataset.")
    print("All fine-tuned models will be saved to your local machine's download folder.")

    # Get all the checkpoints from the user
    checkpoints = []
    while True:
        checkpoint = input('Enter a HuggingFace Model checkpoints ("done" when finished): ')

        if (checkpoint.lower() == "done"):
            break

        checkpoints.append(checkpoint)

    # Get the output directory from the user
    output_base_dir = ''
    while True:
        output_base_dir = input("Enter an output directory path: ")

        if not os.path.exists(output_base_dir):
            print("Invalid path.")
            continue

        if not os.path.isdir(output_base_dir):
            print("Not a directory.")
            continue

        break

    for checkpoint in checkpoints:
        print(f"Processing checkpoint: {checkpoint}")
        output_dir = os.path.join(output_base_dir, checkpoint + "-fine-tuned")

        if fine_tune_model(checkpoint=checkpoint, output_dir=output_dir):
            # Zip the saved model directory
            shutil.make_archive(output_dir, "zip", output_dir)
            # Download the zipped model to local machine
            files.download(f"{output_dir}.zip")
            print(f"\n\nSuccessfully fine-tuned and saved model from checkpoint: {checkpoint}\n\n")
        else:
            print(f"Skipped checkpoint: {checkpoint}\n\n")

if __name__ == '__main__':
    main()