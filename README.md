# 📰 FastAPI News Classifier API

A lightweight and efficient API to classify news articles into categories like **Politics**, **Technology**, **Health**, and **Sports** using **scikit-learn** and **FastAPI**. This project is fully containerized with **Docker** and requires no frontend — just a clean backend API.

---

## ✨ Features

* Preprocessing and keyword-based labeling of news data
* Scikit-learn model using TF-IDF + Multinomial Naive Bayes
* API built with FastAPI to predict category from title & description
* Docker support for easy deployment
* No frontend – just send requests and get predictions

---

## 🧑‍🧠 Model Training

1. **Clone the repo:**

   ```bash
   git clone https://github.com/your-username/fastapi-news-classifier.git
   cd fastapi-news-classifier
   ```

2. **Install dependencies:**
   Make sure you have Python 3.8+ and pip installed.

   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model:**
   This will:

   * Load the dataset (`data.csv`)
   * Clean and preprocess the text
   * Automatically label it with basic rules
   * Train a TF-IDF vectorizer and a Multinomial Naive Bayes model
   * Save the model and vectorizer to disk

   Run:

   ```bash
   python train_model.py
   ```
