# Comment Toxicity Classification

A notebook-based NLP project for multi-label comment classification with TensorFlow/Keras, a bidirectional LSTM, and a Gradio demo.

[Portfolio](https://odra-bira.github.io/) · [Notebook](Toxicity.ipynb)

## What it does

Predicts six independent labels: toxic, severe toxic, obscene, threat, insult, and identity hate. A comment can belong to more than one category.

## Pipeline

1. Load the Jigsaw training CSV.
2. Convert comments into integer sequences with `TextVectorization` (up to 200,000 vocabulary tokens; sequence length 1,800).
3. Apply a 32-dimensional embedding and a bidirectional LSTM with 32 units.
4. Use dense layers with 128, 256, and 128 units, followed by six sigmoid outputs.
5. Train with Adam and binary cross-entropy, then explore predictions in Gradio.

## Setup

```bash
git clone https://github.com/odra-bira/comment-toxicity-clean.git
cd comment-toxicity-clean
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter notebook Toxicity.ipynb
```

On Windows, activate with `.venv\Scripts\activate`.

Obtain the Jigsaw Toxic Comment Classification Challenge data under its usage terms. The dataset and trained weights are not included. Place the extracted `train.csv` at:

```text
data/train.csv
```

Run the notebook in order. The training cell currently runs one epoch. Training may require substantial memory because of the long sequences and in-memory dataset cache. Gradio launches locally by default; stop the notebook kernel to end the session.

## Repository guide

| File | Purpose |
| --- | --- |
| `Toxicity.ipynb` | Preprocessing, model, training, evaluation, and demo |
| `requirements.txt` | Direct Python dependencies |
| `.gitignore` | Excludes data, model weights, environments, and generated files |

## Evaluation limitations

This is an experimental notebook, not a validated moderation service. The current workflow adapts the vocabulary before splitting and derives subsets from a reshuffling dataset; it does not establish a reliable held-out benchmark. Its categorical accuracy calculation is also unsuitable for independent multi-label predictions.

Before reporting performance, split examples deterministically before preprocessing, adapt the vocabulary only on training text, and evaluate per-label precision, recall, F1, and PR-AUC on an untouched test set. No verified accuracy claim is made here.

Predictions can reflect dataset bias and miss context. Human review is needed for consequential moderation decisions.

## Author

Odra Bira · [Portfolio](https://odra-bira.github.io/)
