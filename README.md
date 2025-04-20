# Walmart-Amazon Product Matching

This repository contains code and resources for a product matching project using datasets from Walmart and Amazon. The task involves identifying whether pairs of products from the two retailers refer to the same real-world item. The goal is to train a machine learning model that predicts a match (1) or non-match (0) for each product pair, maximizing the mean F1-score.

## 📁 Files

- `CS252.py`: Main Python script for preprocessing, training, and prediction.
- `submission.csv`: CSV file formatted for evaluation with `id,label`.
- `data/` (not included here): Contains `ltable.csv`, `rtable.csv`, `train.csv`, and `test.csv`.

## 🧠 Task Overview

- **Dataset Description**:

  - `ltable`: Walmart products
  - `rtable`: Amazon products
  - `train.csv`: (ltable_id, rtable_id, label, id)
  - `test.csv`: (ltable_id, rtable_id, id)

- **Goal**: Predict matches between `ltable` and `rtable` pairs.
- **Evaluation Metric**: Mean F1-score.

## 📈 Sample Scores

Submissions made during development and their respective F1-scores:

| Submission | Score   |
| ---------- | ------- |
| v1         | 0.79385 |
| v2         | 0.77777 |
| v3         | 0.79302 |
| v4         | 0.79039 |

## ✅ Submission Format

The expected format for the submission file:

```csv
id,label
1,1
2,0
...
```
