# Test the model myself
import numpy as np
from transformers import (
  TFAutoModelForSequenceClassification,
  create_optimizer,
  DataCollatorWithPadding,
  AutoTokenizer
)

checkpoint = 'distilbert-base-uncased'

# Import model and tokenizer
model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Sample inputs
texts = [
    "I love this movie! It's fantastic.",
    "I hate this movie. It was terrible.",
    "The movie was okay, not great but not bad either.",
    "Hey, this coffee was fantastic.",
    "Hey, this coffee was mid.",
    "How do I do that?"
]

# Tokenize the inputs
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="tf")

outputs = model(inputs) # Pass the inputs through the model
logits = outputs.logits # Get the logits
predictions = np.argmax(logits, axis=1) # Select the ids with highest value in each logit

# maps ids to labels
id2label = {
    0: 'NEGATIVE',
    1: 'NEUTRAL',
    2: 'POSITIVE'
}

# Convert the ids to their labels
predicted_labels = [id2label[label_id] for label_id in predictions]

# Display the results
for text, label in zip(texts, predicted_labels):
  print(f"Text: {text}\nSentiment: {label}\n")