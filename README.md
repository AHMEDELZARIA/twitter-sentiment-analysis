# 🐦 Twitter Sentiment Analysis

## 📋 Overview
This project is designed to fine-tune pre-trained models from the Hugging Face Hub on the Cardiff Twitter Sentiment Datasets using TensorFlow and the 😀Transformers library. The `finetuning.py` script handles the fine-tuning process, while `inference.py` allows users to test their fine-tuned models.

## ✨ Features
- **🔍 Model Selection**: Fine-tune any pre-trained model from the Hugging Face Hub.
- **📊 Dataset**: Utilizes the Cardiff Twitter Sentiment Datasets for training.
- **🧠 TensorFlow Integration**: Leverages TensorFlow for model training.
- **🧪 Inference**: Provides an easy way to test the fine-tuned models.

## ⚙️ Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/twitter-sentiment-analysis.git
    cd twitter-sentiment-analysis
    ```

2. Install dependencies:
    ```sh
    pip install datasets evaluate transformers[sentencepiece]
    pip install tensorflow
    ```

## 🛠️ Fine-Tuning Process

### 1️⃣ Step 1: Model Selection
Choose a pre-trained model from the Hugging Face Hub. The `finetuning.py` script allows you to specify the model name, which will be loaded and prepared for fine-tuning.

### 2️⃣ Step 2: Data Preparation
The script loads the Cardiff Twitter Sentiment Datasets, splits them into training and validation sets, and tokenizes the text data using the selected tokenizer from the Hugging Face library.

### 3️⃣ Step 3: Model Fine-Tuning
The pre-trained model is fine-tuned on the prepared dataset. This involves training the model on the Twitter sentiment data, adjusting the model weights to improve its performance on sentiment classification tasks.

### 4️⃣ Step 4: Evaluation
During fine-tuning, the model's performance is evaluated on the validation set to monitor its accuracy and adjust training parameters as needed.

## 🚀 Usage

### 🔧 Fine-Tuning
Run the `finetuning.py` script with the desired model:
```sh
python finetuning.py
```
You will then be prompted to specify the set of models to train (please provide valid HuggingFace checkpoints from their model hub) as well as the output directory to which the fine-tuned models will be stored.

### 🔍 Inference
Use the `inference.py` script to test the fine-tuned model on some sample tweets:
```sh
python inference.py
```
Feel free to test your models on any new set of tweets to evaluate the performance.

## 📜 Scripts

### `finetuning.py`
Handles the entire fine-tuning process:
- Loads the pre-trained model and tokenizer.
- Prepares the dataset.
- Fine-tunes the model on the Cardiff Twitter Sentiment Datasets.
- Saves the fine-tuned model for inference.

### `inference.py`
Allows for testing the fine-tuned model on new input data:
- Loads the fine-tuned model and tokenizer.
- Processes the input text.
- Outputs the sentiment prediction.

## 🤝 Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss improvements or report bugs.

## 🙏 Acknowledgements
- Hugging Face for the pre-trained models and 😀Transformers library.
- Cardiff University for the Twitter Sentiment Datasets.

Feel free to suggest any additional changes!
