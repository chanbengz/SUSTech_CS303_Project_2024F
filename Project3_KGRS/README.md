# Project3 Knowledge Graph-based Recommendation System

## Description

This project requires us to tune the hyperparameters of the model to achieve the best performance. We will use the knowledge graph-based recommendation system to recommend the click-through rate and top-k movies for a given user. We have two subtasks in this project:
 - **CTR:** Maximum the AUC metric of your score function $f(u,w)$ on a test dataset $Y_{test}$.
 - **Tok-k:** Maximum the nDCG@5 metric of your score function $f(u,w,)$ on a test dataset $Y_{test}$.

## Usage

Install dependencies and train the model

```bash
pip install -r requirements.txt
python src/kgrs.py
```

Evaluate the model with

```bash
python src/evaluate.py
```
