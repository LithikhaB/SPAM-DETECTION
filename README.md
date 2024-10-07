# Spam Detection using Machine Learning

**Classify SMS messages as spam or not spam using a machine learning model trained on a dataset of labeled SMS messages.**

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Dataset](#dataset)
- [License](#license)

## Introduction

This project is a simple web application built using **Streamlit** and **NLTK** that classifies SMS messages as spam or not spam using a machine learning model. The model is trained on a dataset of labeled SMS messages and utilizes natural language processing techniques to predict the likelihood of a message being spam.

## Features

- Classify SMS messages as spam or not spam
- Utilizes a machine learning model trained on a dataset of labeled SMS messages
- Employs natural language processing techniques for text analysis
- Provides a user-friendly web interface using Streamlit

## Requirements

- Python 3.8+
- Streamlit
- NLTK
- Scikit-learn
- Pickle

## Installation

To install the required packages, run the following command:

```bash
pip install streamlit nltk scikit-learn
```

Note: The `pickle` module is included with Python, so no additional installation is needed.

## Usage

To run the application, navigate to the project directory in your terminal and execute the following command:

```bash
streamlit run app.py
```

This command will launch the web application in your default web browser at `http://localhost:8501`.

## Model Training

The machine learning model is trained on a dataset of labeled SMS messages using Scikit-learn. The training process involves the following steps:

1. **Load the dataset**: Collect SMS messages labeled as spam or not spam.
2. **Preprocess the data**: Use NLTK for tokenization, stop word removal, and stemming.
3. **Feature extraction**: Convert text into numerical features using `CountVectorizer`.
4. **Split the dataset**: Divide the dataset into training and testing sets.
5. **Train the model**: Use a classification algorithm (e.g., Naive Bayes, Logistic Regression) on the training data.
6. **Evaluate the model**: Assess the model's performance using metrics like accuracy and F1-score.

### Example Code for Model Training

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Load dataset
data = pd.read_csv('spam.csv')

# Preprocessing
# Here you can include text preprocessing steps (tokenization, stop word removal, etc.)

# Feature extraction
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['message'])  # Text messages
y = data['label']  # Labels (spam or not spam)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = MultinomialNB()
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

## Dataset

The dataset used for training the model is a collection of labeled SMS messages. The dataset is not included in this repository, but you can use your own dataset or a publicly available dataset, such as the [SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.
