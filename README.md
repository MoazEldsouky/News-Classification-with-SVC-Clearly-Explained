# 📰🔍 News Categorizer 98 🎯✨

A Machine Learning-powered web app for classifying news articles into five categories: **Sport, Tech, Entertainment, Politics, and Business**. The project leverages an SVM classifier trained on BBC News data, using a **TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer** for feature extraction, and achieves an impressive **98% accuracy**.

---

## Table of Contents

- [Demo](#demo)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Project Structure](#project-structure)
- [Author](#author)
- [License](#license)
- [References](#references)

---

## Demo

- **Hugging Face Space:**  
  Try the app live: [News Categorizer on Hugging Face Spaces](https://huggingface.co/spaces/moazx/News_Categorizer)

- **Model Training & Explanation:**  
  See the full training process and explanation in the Kaggle notebook:  
  [News Classification with SVC - Clearly Explained](https://www.kaggle.com/code/moazeldsokyx/news-classification-with-svc-clearly-explained)

---

## Features

- 📰 Classifies news articles into 5 categories
- ⚡ Fast, interactive web interface (Gradio)
- 🧹 Automatic text preprocessing (lowercasing, stopword & punctuation removal)
- 📈 SVM model with 98% accuracy
- 🏷️ Label encoding for category mapping
- 📦 Easy to run locally

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MoazEldsouky/News-Classification-with-SVC-Clearly-Explained.git
   cd News-Classification-with-SVC-Clearly-Explained
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data (first run only):**
   The app will automatically download required NLTK data (`punkt`, `stopwords`) on first launch.

---

## Usage

Simply run the following command:

```bash
python app.py
```

This will launch a Gradio web interface in your browser. Paste any news article, and the app will predict its category.

---

## Model Details

- **Algorithm:** Support Vector Machine (SVM)
- **Vectorizer:** **TF-IDF (Term Frequency-Inverse Document Frequency)**
- **Label Encoder:** Maps predicted class indices to category names
- **Preprocessing:** Lowercasing, tokenization, stopword & punctuation removal (using NLTK)
- **Training Data:** BBC News articles ([see Kaggle Notebook](https://www.kaggle.com/code/moazeldsokyx/news-classification-with-svc-clearly-explained))
- **Accuracy:** 98%

---

## Project Structure

```
.
├── app.py                    # Main Gradio app
├── requirements.txt          # Python dependencies
├── SVM_model.joblib          # Trained SVM model
├── vectorizer.joblib         # Fitted vectorizer
├── label_encoder.joblib      # Label encoder for categories
├── preprocess_text_function.pkl # (Optional) Preprocessing function
└── README.md                 # Project documentation
```
---


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---


## Acknowledgements

- BBC News for the dataset
- Gradio for the web UI
- NLTK for text preprocessing
- scikit-learn for machine learning tools

---

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## Issues

If you encounter any problems, please open an issue on GitHub.
