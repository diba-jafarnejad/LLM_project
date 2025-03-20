# LLM Project

## Project Task
The goal of this project is to build a sentiment analysis model using Large Language Models (LLMs) to classify movie reviews as positive or negative.
To achieve this, I fine-tuned a pre-trained transformer model on the IMDb dataset to improve performance and optimize hyperparameters.

## Dataset
I used the IMDb Large Movie Review Dataset, which contains:

50,000 movie reviews labeled as either positive (1) or negative (0).
Balanced dataset (25,000 positive, 25,000 negative).
The dataset was preprocessed by removing punctuation, HTML tags, stopwords, and applying tokenization.

## Pre-trained Model
We fine-tuned the **`distilbert-base-uncased-finetuned-sst-2-english`** model from Hugging Face for sentiment analysis.  

✅ Why distilbert-base-uncased?

Smaller & faster than BERT (40% fewer parameters, 60% faster).
Retains 97% performance of BERT.
Ideal for sentiment analysis, as shown in research papers.

🔗 **Model Link:** [DistilBERT SST-2 on Hugging Face](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)

## Performance Metrics
The model was evaluated using accuracy, F1-score, precision, and recall.

Metric	Score
Accuracy	88.2%
F1-score	88.0%
Precision	88.1%
Recall	87.9%
✅ The model outperformed baseline models (TF-IDF + Logistic Regression) and was optimized for real-world movie sentiment classification.
## Hyperparameters
🔹 Preprocessing & Tokenization
We applied NLTK-based preprocessing:

Lowercasing, removing punctuation, stopwords, and HTML tags.
Tokenization using Hugging Face's AutoTokenizer.
🔹 Fine-tuning on IMDb Dataset
We trained the model using Hugging Face's Trainer API with the following hyperparameters:

Hyperparameter	Value	Why?
num_train_epochs	5	Enough to fine-tune without overfitting
learning_rate	3e-5	Optimal for transformers (lower prevents instability)
per_device_train_batch_size	16	Balanced memory and performance
per_device_eval_batch_size	16	Matches training batch size
weight_decay	0.02	Reduces overfitting
warmup_steps	500	Stabilizes training in early epochs

