# Mental Health Sentiment Analysis using NLP & Machine Learning

This project applies Natural Language Processing (NLP) and Machine Learning techniques to classify mental health–related text into **Depression** or **Normal** categories. Two models — a Decision Tree and a Neural Network — are implemented and compared to evaluate their effectiveness in identifying distress signals from text data.

---

## 📌 Project Objectives
- Preprocess and analyze mental health text data
- Build binary sentiment classification models
- Compare traditional ML and Neural Network performance
- Evaluate model generalization using accuracy and ROC-AUC
- Test models on unseen text statements

---

## 🗂 Dataset Overview
- ~53,000 text records
- Original labels include Anxiety, Depression, Stress, Suicidal, etc.
- Converted into binary classes:
  - **Normal**
  - **Depression** (all non-normal categories)

**Class distribution after binarization:**
- Depression: 36,668
- Normal: 16,379

---

## 🔄 Data Preprocessing
- Missing value handling
- Text cleaning and normalization
- TF-IDF vectorization (max_features = 10,000)
- Label encoding
- 80:20 train-test split

---

## 🧠 Models Implemented
### Decision Tree Classifier
- Simple and interpretable baseline
- Faster training
- More sensitive to feature splits

### Neural Network
- Captures more complex text patterns
- Better class separation
- Higher robustness on unseen data

---

## 📊 Model Performance

### Accuracy Comparison
| Model | Train Accuracy | Test Accuracy |
|------|---------------|---------------|
| Decision Tree | 99.94% | 93.66% |
| Neural Network | 99.94% | 93.66% |

Both models show mild overfitting, but the generalization gap remains under 5%.

---

### ROC–AUC Score
| Model | AUC |
|------|-----|
| Decision Tree | 0.89 |
| Neural Network | **0.93** |

The Neural Network demonstrates superior class separation and overall reliability.

---

## 🧪 Testing on New Statements
The models were tested on unseen text inputs simulating real user expressions.

**Observations:**
- Decision Tree tends to underestimate distress
- Neural Network is more sensitive but may overclassify negative sentiment
- Both models capture emotional tone but lack deep contextual understanding

---

## ⚠️ Limitations
- Binary classification oversimplifies mental health conditions
- TF-IDF lacks semantic understanding
- Class imbalance affects predictions
- Contextual nuance is not fully captured

---

## 🚀 Future Improvements
- Use word embeddings (Word2Vec, GloVe)
- Fine-tune transformer models (BERT, RoBERTa)
- Apply data augmentation techniques
- Experiment with ensemble models

---

## 🛠️ Tech Stack
- Python
- Scikit-learn
- TensorFlow / Keras
- Pandas, NumPy
- Jupyter Notebook

---

## ▶️ How to Run
1. Clone the repository
   ```bash
   git clone https://github.com/your-username/mental-health-sentiment-analysis.git
