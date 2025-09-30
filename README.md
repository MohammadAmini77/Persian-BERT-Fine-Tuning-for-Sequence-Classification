# Persian BERT Fine-Tuning for Sequence Classification

This repository provides a Jupyter Notebook for **fine-tuning Persian BERT models** on a sequence classification task (e.g., sentiment analysis, topic classification, or other NLP tasks in Persian).  
It demonstrates how to leverage [Hugging Face Transformers](https://huggingface.co/transformers/) to train, evaluate, and test a pre-trained Persian BERT model.

---

## 📋 Features
- Load and preprocess Persian text datasets.
- Tokenize text using `bert-base-parsbert-uncased` (or any Persian-compatible BERT model).
- Fine-tune a BERT model for classification.
- Evaluate model performance using accuracy, F1-score, and confusion matrix.
- Save and load trained models for inference.

---

## 🗂️ Repository Structure
```
.
├── Persian-BERT-Fine-Tuning-for-Sequence-Classification.ipynb  # Main notebook
└── README.md                                                   # Project documentation
```

---

## ⚡ Requirements
Install the required Python libraries before running the notebook:

```bash
pip install transformers datasets scikit-learn pandas numpy matplotlib
```

Optional for GPU acceleration:
```bash
pip install torch --extra-index-url https://download.pytorch.org/whl/cu118
```

---

## 🚀 Usage

1. **Clone this repository**
```bash
git clone https://github.com/<your-username>/Persian-BERT-Fine-Tuning.git
cd Persian-BERT-Fine-Tuning
```

2. **Open the notebook**  
Launch Jupyter or Colab:
```bash
jupyter notebook Persian-BERT-Fine-Tuning-for-Sequence-Classification.ipynb
```
or upload the notebook to [Google Colab](https://colab.research.google.com/).

3. **Run the cells**
- Upload or load your Persian dataset.
- Adjust hyperparameters (batch size, learning rate, epochs).
- Run training and evaluation.

---

## 📝 Example Dataset
The notebook supports any CSV/TSV dataset with columns such as:

| text | label |
|------|------|
| این فیلم عالی بود | 1 |
| خیلی خسته کننده بود | 0 |

Update the dataset loading cell to match your file path and column names.

---

## 📊 Evaluation
The notebook outputs:
- Accuracy
- Precision, Recall, F1-Score
- Confusion matrix visualization

---

## 💡 Tips
- Use a GPU for faster training (e.g., Colab GPU runtime).
- Experiment with different Persian BERT models from [Hugging Face](https://huggingface.co/models?language=fa).
- For larger datasets, consider mixed precision training (`fp16`) to speed up training.

---

## 📚 References
- [ParsBERT](https://huggingface.co/HooshvareLab/bert-base-parsbert-uncased)
- [Transformers Documentation](https://huggingface.co/docs/transformers/index)
- [Dataset]("persiannlp/parsinlu_query_paraphrasing")


---

