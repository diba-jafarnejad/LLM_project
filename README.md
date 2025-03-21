# LLM Project

## Project Task
The goal of this project is to build a sentiment analysis model using Large Language Models (LLMs) to classify movie reviews as positive or negative.
To achieve this, I fine-tuned a pre-trained transformer model on the IMDb dataset to improve performance and optimize hyperparameters.

## Dataset
The dataset used is the **IMDb Large Movie Review Dataset**, containing:
- **50,000 reviews** labeled as **positive (1) or negative (0)**.
- A **balanced dataset** (25,000 positive, 25,000 negative reviews).

## Pre-trained Model
I used distilbert-base-uncased-finetuned-sst-2-english, a lightweight and efficient transformer model, specifically trained for sentiment analysis.

✅ Why did I choose this model?

It was pre-trained on the Stanford Sentiment Treebank (SST-2) dataset.
It is optimized for binary sentiment classification.
It is lighter and faster than BERT, making it ideal for my project.
🔗 Model Link: DistilBERT SST-2 on Hugging Face

## Performance Metrics
The model was evaluated using accuracy, F1-score, precision, and recall.

| Metric     | Score  |
|------------|--------|
| Accuracy   | 88.2%  |
| F1-score   | 88.0%  |
| Precision  | 88.1%  |
| Recall     | 87.9%  |

✅ The model outperformed baseline models (TF-IDF + Logistic Regression) and was optimized for real-world movie sentiment classification.

## Hyperparameters
🔹 Preprocessing & Tokenization
I applied NLTK-based preprocessing:
Lowercasing, removing punctuation, stopwords, and HTML tags.
Tokenization using Hugging Face's AutoTokenizer.
🔹 Fine-tuning on IMDb Dataset
I trained the model using Hugging Face's Trainer API with the following hyperparameters:

Optimizing these hyperparameters improved accuracy, generalization, and efficiency while preventing overfitting or instability.
This fine-tuning allowed my model to achieve 88.2% accuracy on IMDb reviews, making it highly effective for sentiment classification.

Hyperparameter		                             Value                                    Why?
num_train_epochs	                              5	                                      Enough to fine-tune without overfitting
learning_rate	                                  3e-5	                                  Optimal for transformers (lower prevents instability)
per_device_train_batch_size	                    16                                      Balanced memory and performance
per_device_eval_batch_size	                    16	                                    Matches training batch size
weight_decay	                                  0.02	                                  Reduces overfitting
warmup_steps	                                  500	                                    Stabilizes training in early epochs

## Relevant Links

🚀 [imdb dataset on Hugging Face](https://huggingface.co/datasets/stanfordnlp/imdb)

🚀 [View My Fine-Tuned Model on Hugging Face](https://huggingface.co/dibajafarnejad/imdb-optimized-finetuned-distilbert/tree/main)
